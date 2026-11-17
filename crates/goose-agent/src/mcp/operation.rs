use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use goose_provider_types::conversation::message::{Message, ToolRequest};
use goose_provider_types::conversation::Conversation;
use rmcp::model::{CallToolRequestParams, Tool};

use super::{
    get_tool_owner, recover_mangled_tool_name, CallRequest, ExtensionLease, ExtensionManager,
    ExtensionSet, ExtensionSpec, ToolCallBatch, ToolCallResult,
};
use crate::operation::{
    applied, messages_since_kickoff, not_applicable, Emitter, Operation, OperationResult,
};
use crate::tool_call_state::{tool_calls, ToolCallState};
use crate::tool_visibility::is_tool_visible_to_model;

/// Chooses the extensions a session runs with.
#[async_trait]
pub trait ExtensionSource<S, C: ExtensionSpec>: Send + Sync {
    async fn extensions(&self, session: &S) -> Result<ExtensionSet<C>>;
}

/// Advertises the tools of a session's MCP extensions and runs the calls the
/// model makes to them.
///
/// Place it after [`ToolApprovalOperation`](crate::tool_approval::ToolApprovalOperation):
/// it only runs calls that approval marked executable. Calls to tools it does
/// not own are left for later operations. Each collection or execution acquires
/// a lease from the current session; inference leases are not held across approval.
pub struct ExtensionOperation<S, C: ExtensionSpec> {
    manager: Arc<ExtensionManager<C>>,
    source: Arc<dyn ExtensionSource<S, C>>,
    batch: ToolCallBatch,
}

impl<S, C: ExtensionSpec> ExtensionOperation<S, C> {
    pub fn new(manager: Arc<ExtensionManager<C>>, source: Arc<dyn ExtensionSource<S, C>>) -> Self {
        Self {
            manager,
            source,
            batch: ToolCallBatch::new(),
        }
    }

    async fn lease(&self, session: &S) -> Result<ExtensionLease<C>> {
        let set = self.source.extensions(session).await?;
        Ok(self.manager.resolve(&set).await)
    }
}

fn executable_requests(conversation: &Conversation) -> Result<Vec<ToolRequest>> {
    Ok(tool_calls(messages_since_kickoff(conversation)?)
        .into_iter()
        .filter(|(request, state)| {
            *state == ToolCallState::Executable
                && request.tool_call.is_ok()
                && !request.was_executed_externally()
        })
        .map(|(request, _)| request.clone())
        .collect())
}

fn owned_call(request: &ToolRequest, tools: &[Tool]) -> Option<CallToolRequestParams> {
    let mut call = request.tool_call.as_ref().ok()?.clone();
    if tools.iter().any(|tool| tool.name == call.name) {
        return Some(call);
    }
    let owners: Vec<_> = tools.iter().map(get_tool_owner).collect();
    let recovered = recover_mangled_tool_name(
        &call.name,
        tools
            .iter()
            .zip(&owners)
            .map(|(tool, owner)| (tool.name.as_ref(), owner.as_deref())),
    )?;
    call.name = recovered.into();
    Some(call)
}

#[async_trait]
impl<S, C, E> Operation<S, E> for ExtensionOperation<S, C>
where
    S: Send + Sync + 'static,
    C: ExtensionSpec,
    E: From<Message> + Send + 'static,
{
    fn name(&self) -> &'static str {
        "extensions"
    }

    async fn finalize_cancellation(
        &self,
        _session: &S,
        _conversation: &Conversation,
        emit: &Emitter,
    ) -> Vec<E> {
        let (actions, response) = self.batch.take();
        actions
            .into_iter()
            .chain(response.map(|response| emit.message(response)))
            .map(E::from)
            .collect()
    }

    async fn inference_tools(&self, session: &S) -> Result<Vec<Tool>> {
        let lease = self.lease(session).await?;
        Ok(lease
            .tools()
            .await
            .into_iter()
            .filter(is_tool_visible_to_model)
            .collect())
    }

    async fn run(
        &self,
        session: &S,
        conversation: &Conversation,
        emit: &Emitter,
    ) -> Result<OperationResult<E>> {
        let requests = executable_requests(conversation)?;
        if requests.is_empty() {
            return not_applicable();
        }
        let lease = self.lease(session).await?;
        let tools = lease.tools().await;
        let owned: Vec<_> = requests
            .into_iter()
            .filter_map(|request| owned_call(&request, &tools).map(|call| (request, call)))
            .collect();
        if owned.is_empty() {
            return not_applicable();
        }

        let mut calls = Vec::with_capacity(owned.len());
        for (request, call) in owned {
            let result = lease
                .call(
                    call,
                    CallRequest::new(request.id.clone()),
                    emit.cancel_token().clone(),
                )
                .await
                .unwrap_or_else(|error| ToolCallResult::from(Err(error)));
            calls.push((request, result));
        }
        let (actions, response) = self.batch.run(calls, emit).await?;
        applied(
            actions
                .into_iter()
                .chain([emit.message(response)])
                .map(E::from),
        )
    }
}
