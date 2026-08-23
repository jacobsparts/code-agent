try:
    from . import cursor_codec as _codec
except ImportError:
    import cursor_codec as _codec

F = _codec.F
UNKNOWN = _codec.UNKNOWN

# Underscore-prefixed names are structural designations not supplied by cursor.py.
SUPPLEMENTAL_MESSAGES = {
    'BackgroundShellSpawnArgs': [
        F('command', 1, type='string'),
        F('workingDirectory', 2, type='string'),
        F('toolCallId', 3, type='string'),
        F('parsingResult', 4, type='bytes'),
        F('sandboxPolicy', 5, type='bytes'),
        F('enableWriteShellStdinTool', 6, type='bool'),
        F('description', 7, type='string'),
        F('classifierResult', 8, type='bytes'),
        F('outputNotification', 9, type='bytes'),
        F('smartModeApproval', 10, type='bytes'),
        F('hookApprovalRequirement', 11, type='bytes'),
        F('skipApproval', 12, type='bool'),
        F('conversationId', 13, type='string'),
    ],
    'BidiAppendRequest': [
        F('payloadHex', 1, type='string'),
        F('requestId', 2, type='M:BidiRequestId'),
        F('appendSeqno', 3, type='int64'),
        F('payload', 4, type='bytes'),
    ],
    # agent.v1.AgentService.Run request envelope: the codec leaves Run_req empty
    # because captures show five distinct top-level variants (oneof members).
    'Run_req': [
        F('runRequest', 1, type='M:RunRequest'),
        F('execClientMessage', 2, type='M:ExecClientMessage'),
        F('kvClientMessage', 3, type='M:KvClientMessage'),
        F('execClientControlMessage', 5, type='M:ExecClientControlMessage'),
        F('clientHeartbeat', 7, type='M:_ClientHeartbeat'),
    ],
    '_ClientHeartbeat': [],
    'BidiRequestId': [
        F('requestId', 1, type='string'),
    ],
    'CanvasDiagnosticsArgs': [
        F('path', 1, type='string'),
        F('toolCallId', 2, type='string'),
    ],
    'ComputerUseArgs': [
        F('toolCallId', 1, type='string'),
        F('actions', 2, type='bytes'),
    ],
    'ConversationSearchArgs': [
        F('query', 1, type='string'),
        F('toolCallId', 2, type='string'),
        F('limit', 3, type='int32'),
    ],
    'DeleteArgs': [
        F('path', 1, type='string'),
        F('toolCallId', 2, type='string'),
    ],
    'DiagnosticsArgs': [
        F('path', 1, type='string'),
        F('toolCallId', 2, type='string'),
    ],
    'ExecuteHookArgs': [
        F('request', 1, type='bytes'),
    ],
    'FetchArgs': [
        F('url', 1, type='string'),
        F('toolCallId', 2, type='string'),
    ],
    'ForceBackgroundShellArgs': [
        F('toolCallId', 1, type='string'),
    ],
    'ForceBackgroundSubagentArgs': [
        F('toolCallId', 1, type='string'),
    ],
    'GetBlobArgs': [
        F('blobId', 1, type='bytes'),
    ],
    'GetBlobResult': [
        F('_field1', 1, type='bytes'),
    ],
    'GetDiffRequest': [
        F('cwd', 1, type='string'),
        F('ref', 2, type='string'),
        F('baseRef', 3, type='string'),
        F('mergeBase', 4, type='bool'),
        F('targetPaths', 5, type='string', rep=True),
        F('unifiedContextLines', 6, type='int32'),
        F('maxUntrackedFiles', 7, type='int32'),
        F('outputFormat', 8, type='int32'),
        F('submoduleRecurseDepth', 9, type='int32'),
        F('includeSpaceChanges', 10, type='bool'),
        F('committedOnly', 11, type='bool'),
        F('computePatchId', 12, type='bool'),
        F('returnHeadSha', 13, type='bool'),
        F('maxResponseBytes', 14, type='int32'),
    ],
    'GrepArgs': [
        F('pattern', 1, type='string'),
        F('path', 2, type='string'),
        F('glob', 3, type='string'),
        F('outputMode', 4, type='string'),
        F('contextBefore', 5, type='int32'),
        F('contextAfter', 6, type='int32'),
        F('context', 7, type='int32'),
        F('caseInsensitive', 8, type='bool'),
        F('type', 9, type='string'),
        F('headLimit', 10, type='int32'),
        F('multiline', 11, type='bool'),
        F('sort', 12, type='string'),
        F('sortAscending', 13, type='bool'),
        F('toolCallId', 14, type='string'),
        F('sandboxPolicy', 15, type='bytes'),
        F('offset', 16, type='int32'),
    ],
    'ListMcpResourcesExecArgs': [
        F('server', 1, type='string'),
    ],
    'LsArgs': [
        F('path', 1, type='string'),
        F('ignore', 2, type='string'),
        F('toolCallId', 3, type='string'),
        F('sandboxPolicy', 4, type='bytes'),
        F('timeoutMs', 5, type='int32'),
    ],
    'McpAllowlistPrecheckArgs': [
        F('providerIdentifier', 1, type='string'),
        F('toolName', 2, type='string'),
        F('toolCallId', 3, type='string'),
    ],
    'McpArgs': [
        F('name', 1, type='string'),
        F('arguments', 2, map=('string', 'M:_Value')),
        F('toolCallId', 3, type='string'),
        F('providerIdentifier', 4, type='string'),
        F('toolName', 5, type='string'),
        F('smartModeApproval', 6, type='bytes'),
        F('smartModeApprovalOnly', 7, type='bool'),
        F('skipApproval', 8, type='bool'),
        F('serverIdentifier', 9, type='string'),
    ],
    'McpStateExecArgs': [
        F('serverIdentifiers', 1, type='string', rep=True),
        F('kickOnly', 2, type='bool'),
    ],
    'ModelDetails': [
        F('_field1', 1, type='string'),
        F('_field3', 3, type='string'),
        F('_field4', 4, type='string'),
        F('_field5', 5, type='string'),
        F('_field7', 7, type='bool'),
    ],
    'PiBashExecArgs': [
        F('command', 1, type='string'),
        F('timeout', 2, type='double'),
    ],
    'PiEditExecArgs': [
        F('path', 1, type='string'),
        F('edits', 2, type='bytes'),
    ],
    'PiFindExecArgs': [
        F('pattern', 1, type='string'),
        F('path', 2, type='string'),
        F('limit', 3, type='int32'),
    ],
    'PiGrepExecArgs': [
        F('pattern', 1, type='string'),
        F('path', 2, type='string'),
        F('glob', 3, type='string'),
        F('ignoreCase', 4, type='bool'),
        F('literal', 5, type='bool'),
        F('context', 6, type='int32'),
        F('limit', 7, type='int32'),
    ],
    'PiLsExecArgs': [
        F('path', 1, type='string'),
        F('limit', 2, type='int32'),
    ],
    'PiReadExecArgs': [
        F('path', 1, type='string'),
        F('offset', 2, type='int32'),
        F('limit', 3, type='int32'),
    ],
    'PiWriteExecArgs': [
        F('path', 1, type='string'),
        F('content', 2, type='string'),
    ],
    'PrefetchedBlob': [
        F('blobId', 1, type='bytes'),
        F('value', 2, type='bytes'),
    ],
    'ReadArgs': [
        F('path', 1, type='string'),
        F('toolCallId', 2, type='string'),
        F('offset', 4, type='int32'),
        F('limit', 5, type='int32'),
        F('encodingHint', 6, type='string'),
    ],
    'ReadMcpResourceExecArgs': [
        F('server', 1, type='string'),
        F('uri', 2, type='string'),
        F('downloadPath', 3, type='string'),
        F('toolCallId', 4, type='string'),
        F('smartModeApproval', 5, type='bytes'),
    ],
    'RecordScreenArgs': [
        F('mode', 1, type='int32'),
        F('toolCallId', 2, type='string'),
        F('saveAsFilename', 3, type='string'),
    ],
    'RequestContextArgs': [
        F('notesSessionId', 2, type='string'),
        F('workspaceId', 3, type='string'),
        F('readOnlyPinnedTreeSha', 4, type='string'),
        F('readOnlyPluginCacheRoot', 5, type='string'),
        F('useCached', 7, type='bool'),
    ],
    'ShellAllowlistPrecheckArgs': [
        F('command', 1, type='string'),
        F('workingDirectory', 2, type='string'),
        F('parsingResult', 3, type='bytes'),
        F('classifierResult', 4, type='bytes'),
        F('toolCallId', 5, type='string'),
    ],
    'ShellArgs': [
        F('command', 1, type='string'),
        F('workingDirectory', 2, type='string'),
        F('timeout', 3, type='int32'),
        F('toolCallId', 4, type='string'),
        F('simpleCommands', 5, type='string', rep=True),
        F('hasInputRedirect', 6, type='bool'),
        F('hasOutputRedirect', 7, type='bool'),
        F('parsingResult', 8, type='bytes'),
        F('requestedSandboxPolicy', 9, type='bytes'),
        F('fileOutputThresholdBytes', 10, type='int64'),
        F('isBackground', 11, type='bool'),
        F('skipApproval', 12, type='bool'),
        F('timeoutBehavior', 13, type='int32'),
        F('hardTimeout', 14, type='int32'),
        F('description', 15, type='string'),
        F('classifierResult', 16, type='bytes'),
        F('closeStdin', 17, type='bool'),
        F('outputNotification', 18, type='bytes'),
        F('smartModeApproval', 19, type='bytes'),
        F('hookApprovalRequirement', 20, type='bytes'),
        F('conversationId', 21, type='string'),
    ],
    'SmartModeClassifierArgs': [
        F('toolCallId', 1, type='string'),
        F('parentConversationId', 2, type='string'),
        F('target', 3, type='bytes'),
        F('conversationContext', 4, type='bytes'),
    ],
    'SubagentArgs': [
        F('toolCallId', 1, type='string'),
        F('subagentType', 2, type='string'),
        F('modelId', 3, type='string'),
        F('prompt', 4, type='string'),
        F('readonly', 5, type='bool'),
        F('resumeAgentId', 6, type='string'),
        F('runInBackground', 7, type='bool'),
        F('continuationConfig', 8, type='bytes'),
        F('parentConversationId', 9, type='string'),
        F('apiKeyCredentials', 10, type='bytes'),
        F('azureCredentials', 11, type='bytes'),
        F('bedrockCredentials', 12, type='bytes'),
        F('interrupt', 13, type='bool'),
        F('mode', 14, type='int32'),
        F('forkAgentId', 15, type='string'),
        F('rootParentConversationId', 16, type='string'),
        F('selectedContext', 17, type='bytes'),
        F('directMetaParentChildSubagent', 18, type='bool'),
        F('environment', 19, type='int32'),
        F('cloudBaseBranch', 20, type='string'),
    ],
    'SubagentAwaitArgs': [
        F('agentId', 1, type='string'),
        F('timeoutMs', 2, type='int32'),
    ],
    'WebFetchAllowlistPrecheckArgs': [
        F('url', 1, type='string'),
        F('toolCallId', 2, type='string'),
    ],
    'WriteArgs': [
        F('path', 1, type='string'),
        F('fileText', 2, type='string'),
        F('toolCallId', 3, type='string'),
        F('returnFileContentAfterWrite', 4, type='bool'),
        F('fileBytes', 5, type='bytes'),
        F('encodingHint', 6, type='string'),
    ],
    'WriteShellStdinArgs': [
        F('shellId', 1, type='int32'),
        F('chars', 2, type='string'),
    ],
    '_ClientControlMessage': [
        F('_field3', 3, type='M:_ClientControlReason'),
    ],
    '_ClientControlReason': [
        F('reason', 1, type='string'),
    ],
    '_ModelDetails': [
        F('_field1', 1, type='string'),
        F('_field3', 3, type='string'),
        F('_field4', 4, type='string'),
        F('_field5', 5, type='string'),
        F('_field7', 7, type='bool'),
    ],
    '_BidiRequestId': [
        F('requestId', 1, type='string'),
    ],
    '_BidiAppendRequest': [
        F('payloadHex', 1, type='string'),
        F('requestId', 2, type='M:_BidiRequestId'),
        F('appendSeqno', 3, type='int64'),
        F('payload', 4, type='bytes'),
    ],
    '_McpTool': [
        F('_field1', 1, type='string'),
        F('_field2', 2, type='string'),
        F('_field4', 4, type='string'),
        F('_field5', 5, type='string'),
        F('_field6', 6, type='string'),
    ],
    '_McpTools': [
        F('_field1', 1, type='M:_McpTool', rep=True),
    ],
    # Supplemental schemas for protobuf Value/Struct/ListValue (google.protobuf.Value).
    # _Value uses google.protobuf.Value field numbers: null_value=1, number_value=2 (double),
    # string_value=3, bool_value=4, struct_value=5, list_value=6.
    '_Value': [
        F('nullValue', 1, type='int32'),
        F('numberValue', 2, type='double'),
        F('stringValue', 3, type='string'),
        F('boolValue', 4, type='bool'),
        F('structValue', 5, type='M:_Struct'),
        F('listValue', 6, type='M:_ListValue'),
    ],
    '_Struct': [
        F('fields', 1, type='M:_StructEntry', rep=True),
    ],
    '_StructEntry': [
        F('key', 1, type='string'),
        F('value', 2, type='M:_Value'),
    ],
    '_ListValue': [
        F('values', 1, type='M:_Value', rep=True),
    ],
    # Filtered usage request/response (aiserver.v1.DashboardService/GetFilteredUsageEvents) - invented, prefix _
    '_FilteredUsageRequest': [
        F('_field1', 1, type='int32'),
        F('startDate', 2, type='int64'),
        F('endDate', 3, type='int64'),
        F('page', 6, type='int32'),
        F('pageSize', 7, type='int32'),
    ],
    '_FilteredUsageResponse': [
        F('events', 3, type='M:_FilteredUsageEvent', rep=True),
    ],
    '_FilteredUsageEvent': [
        F('timestamp', 1, type='int64'),
        F('_field8', 8, type='int64'),
        F('tokenUsage', 9, type='M:_FilteredUsageToken'),
        F('conversationId', 23, type='string'),
    ],
    '_FilteredUsageToken': [
        F('uncachedInput', 1, type='int64'),
        F('outputTokens', 2, type='int64'),
        F('cacheWrite', 3, type='int64'),
        F('cacheRead', 4, type='int64'),
    ],
    # Historical conversation-state tool step (ConversationCheckpointUpdate.turns inner blob) - invented, prefix _
    '_HistoricalToolStep': [
        F('_field2', 2, type='M:_HistoricalToolStepContainer'),
    ],
    '_HistoricalToolStepContainer': [
        F('_field15', 15, type='M:_HistoricalToolStepPayload'),
    ],
    '_HistoricalToolStepPayload': [
        F('_field1', 1, type='M:McpArgs'),
        F('_field2', 2, type='M:_HistoricalToolOutcome'),
    ],
    '_HistoricalToolOutcome': [
        F('_field1', 1, type='M:_HistoricalToolText'),
        F('_field2', 2, type='bool'),
    ],
    '_HistoricalToolText': [
        F('content', 1, type='string'),
    ],
    '_HistoricalToolTextWrapper': [
        F('_field1', 1, type='M:_HistoricalToolText'),
    ],
    # Agent turn wrapper inside ConversationCheckpointUpdate.turns - invented, prefix _
    '_HistoricalTurn': [
        F('_field1', 1, type='bytes'),
        F('_field2', 2, type='bytes', rep=True),
    ],
    # KV response-boundary blob (under Run_res.kvServerMessage.setBlobArgs.blobData) - invented, prefix _
    '_BoundaryBlob': [
        F('structure', 1, type='M:_BoundaryStructure'),
    ],
    '_BoundaryStructure': [
        F('userMessage', 1, type='bytes'),
        F('step', 2, type='bytes', rep=True),
        F('requestId', 3, type='string'),
        F('_field4', 4, type='bytes'),
        F('_field5', 5, type='bool'),
    ],
}

SUPPLEMENTAL_FIELDS = {
    'AgentClientMessage': [
        F('_field4', 4, type='M:_ClientControlMessage'),
    ],
    'ExecServerMessage': [
        F('shellArgs', 2, type='M:ShellArgs'),
        F('writeArgs', 3, type='M:WriteArgs'),
        F('deleteArgs', 4, type='M:DeleteArgs'),
        F('grepArgs', 5, type='M:GrepArgs'),
        F('readArgs', 7, type='M:ReadArgs'),
        F('lsArgs', 8, type='M:LsArgs'),
        F('diagnosticsArgs', 9, type='M:DiagnosticsArgs'),
        F('requestContextArgs', 10, type='M:RequestContextArgs'),
        F('mcpArgs', 11, type='M:McpArgs'),
        F('shellStreamArgs', 14, type='M:ShellArgs'),
        F('backgroundShellSpawnArgs', 16, type='M:BackgroundShellSpawnArgs'),
        F('listMcpResourcesExecArgs', 17, type='M:ListMcpResourcesExecArgs'),
        F('readMcpResourceExecArgs', 18, type='M:ReadMcpResourceExecArgs'),
        F('fetchArgs', 20, type='M:FetchArgs'),
        F('recordScreenArgs', 21, type='M:RecordScreenArgs'),
        F('computerUseArgs', 22, type='M:ComputerUseArgs'),
        F('writeShellStdinArgs', 23, type='M:WriteShellStdinArgs'),
        F('executeHookArgs', 27, type='M:ExecuteHookArgs'),
        F('subagentArgs', 28, type='M:SubagentArgs'),
        F('redactedReadArgs', 29, type='M:ReadArgs'),
        F('forceBackgroundShellArgs', 30, type='M:ForceBackgroundShellArgs'),
        F('forceBackgroundSubagentArgs', 31, type='M:ForceBackgroundSubagentArgs'),
        F('mcpStateExecArgs', 36, type='M:McpStateExecArgs'),
        F('subagentAwaitArgs', 37, type='M:SubagentAwaitArgs'),
        F('smartModeClassifierArgs', 38, type='M:SmartModeClassifierArgs'),
        F('canvasDiagnosticsArgs', 40, type='M:CanvasDiagnosticsArgs'),
        F('shellAllowlistPrecheckArgs', 41, type='M:ShellAllowlistPrecheckArgs'),
        F('mcpAllowlistPrecheckArgs', 42, type='M:McpAllowlistPrecheckArgs'),
        F('webFetchAllowlistPrecheckArgs', 43, type='M:WebFetchAllowlistPrecheckArgs'),
        F('gitDiffRequest', 44, type='M:GetDiffRequest'),
        F('piReadArgs', 45, type='M:PiReadExecArgs'),
        F('piBashArgs', 46, type='M:PiBashExecArgs'),
        F('piEditArgs', 47, type='M:PiEditExecArgs'),
        F('piWriteArgs', 48, type='M:PiWriteExecArgs'),
        F('piGrepArgs', 49, type='M:PiGrepExecArgs'),
        F('piFindArgs', 50, type='M:PiFindExecArgs'),
        F('piLsArgs', 51, type='M:PiLsExecArgs'),
        F('conversationSearchArgs', 53, type='M:ConversationSearchArgs'),
    ],
    'InteractionUpdate': [
        F('textDelta', 1, type='bytes'),
        F('toolCallStarted', 2, type='bytes'),
        F('toolCallCompleted', 3, type='bytes'),
        F('thinkingDelta', 4, type='bytes'),
        F('thinkingCompleted', 5, type='bytes'),
        F('userMessageAppended', 6, type='bytes'),
        F('partialToolCall', 7, type='bytes'),
        F('tokenDelta', 8, type='bytes'),
        F('summary', 9, type='bytes'),
        F('summaryStarted', 10, type='bytes'),
        F('summaryCompleted', 11, type='bytes'),
        F('shellOutputDelta', 12, type='bytes'),
        F('heartbeat', 13, type='bytes'),
        F('turnEnded', 14, type='bytes'),
        F('toolCallDelta', 15, type='bytes'),
        F('stepStarted', 16, type='bytes'),
        F('stepCompleted', 17, type='bytes'),
        F('promptSuggestion', 18, type='bytes'),
        F('postRequestPrompt', 19, type='bytes'),
        F('activeBranchChange', 20, type='bytes'),
        F('feedbackRequest', 21, type='bytes'),
        F('responseComparison', 22, type='bytes'),
    ],
    'KvClientMessage': [
        F('getBlobResult', 2, type='M:GetBlobResult'),
    ],
    'KvServerMessage': [
        F('getBlobArgs', 2, type='M:GetBlobArgs'),
    ],
    'Run_res_ShellArgs': [
        F('workingDirectory', 2, type='string'),
        F('hasInputRedirect', 6, type='bool'),
        F('hasOutputRedirect', 7, type='bool'),
        F('requestedSandboxPolicy', 9, type='bytes'),
        F('isBackground', 11, type='bool'),
        F('skipApproval', 12, type='bool'),
        F('classifierResult', 16, type='bytes'),
        F('outputNotification', 18, type='bytes'),
        F('smartModeApproval', 19, type='bytes'),
        F('hookApprovalRequirement', 20, type='bytes'),
    ],
    'AgentClientMessage_RunRequest': [
        F('modelDetails', 3, type='M:ModelDetails'),
        F('mcpFileSystemOptions', 6, type='bytes'),
        F('skillOptions', 7, type='bytes'),
        F('customSystemPrompt', 8, type='string'),
        F('subagentTypeName', 11, type='string'),
        F('harness', 13, type='string'),
        F('selectedSubagentModelDetails', 15, type='bytes', rep=True),
        F('prefetchedBlobs', 17, type='M:PrefetchedBlob', rep=True),
        F('devRawModelSlug', 18, type='string'),
        F('clientSupportsInlineImages', 19, type='bool'),
        F('subagentModelOverrides', 20, type='bytes', rep=True),
        F('canCreateCloudSubagents', 21, type='bool'),
        F('suppressSubagentProgressUpdateTool', 22, type='bool'),
        F('clientSupportsSendToUser', 23, type='bool'),
    ],
    'AgentClientMessage_UserMessage': [
        F('isSimulatedMsg', 5, type='bool'),
        F('bestOfNGroupId', 6, type='string'),
        F('tryUseBestOfNPromotion', 7, type='bool'),
        F('richText', 8, type='string'),
        F('simulatedMsgReason', 9, type='int32'),
        F('conversationStateBlobId', 10, type='bytes'),
        F('subagentSystemReminder', 11, type='string'),
        F('triggeringUserInfo', 13, type='bytes'),
        F('executePlanInfo', 14, type='bytes'),
        F('simulatedMessageMetadata', 15, type='bytes'),
        F('promptReferenceId', 16, type='string'),
        F('threadId', 17, type='string'),
        F('textBlobId', 18, type='bytes'),
        F('richTextBlobId', 19, type='bytes'),
        F('hookAdditionalContexts', 21, type='bytes', rep=True),
        F('customModeIntent', 22, type='bytes'),
    ],
}

SUPPLEMENTAL_ENDPOINTS = {
    'aiserver.v1.BidiService.BidiAppend [req]': 'BidiAppendRequest',
    # GetFilteredUsageEvents is not in the capture; schemas are invented.
    'aiserver.v1.DashboardService.GetFilteredUsageEvents [req]': '_FilteredUsageRequest',
    'aiserver.v1.DashboardService.GetFilteredUsageEvents [res]': '_FilteredUsageResponse',
}

def merge_messages(primary, supplemental, additions):
    merged = {name: list(fields) for name, fields in supplemental.items()}
    for name, fields in primary.items():
        existing = {field.num: field for field in merged.get(name, ())}
        existing.update({field.num: field for field in fields})
        merged[name] = [existing[number] for number in sorted(existing)]
    for name, fields in additions.items():
        existing = {field.num: field for field in merged.get(name, ())}
        for field in fields:
            existing.setdefault(field.num, field)
        merged[name] = [existing[number] for number in sorted(existing)]
    return merged

MESSAGES = merge_messages(_codec.MESSAGES, SUPPLEMENTAL_MESSAGES, SUPPLEMENTAL_FIELDS)
ENDPOINTS = {**SUPPLEMENTAL_ENDPOINTS, **_codec.ENDPOINTS}
_codec.MESSAGES = MESSAGES
_codec.ENDPOINTS = ENDPOINTS

if hasattr(_codec, "_orig_encode"):
    _orig_encode = _codec._orig_encode
    _orig_decode = _codec._orig_decode
else:
    _orig_encode = _codec.encode
    _orig_decode = _codec.decode
    _codec._orig_encode = _orig_encode
    _codec._orig_decode = _orig_decode

def _encode_unsorted(obj, msg_type):
    msgs = MESSAGES[msg_type]
    by_name = {g.name: g for g in msgs}
    by_name.update({f"f{g.num}": g for g in msgs})
    items = []
    seen = {}
    for k, v in obj.items():
        if v is None:
            continue
        if k == _codec.UNKNOWN:
            for num, occs in v.items():
                n = int(num)
                if n in seen and seen[n] != _codec.UNKNOWN:
                    raise ValueError(f"field {n} of {msg_type} supplied twice: {seen[n]!r} and {_codec.UNKNOWN}")
                seen[n] = _codec.UNKNOWN
                for wt, raw in occs:
                    items.append((n, None, (wt, _codec._enc_varint(raw) if isinstance(raw, int) else raw)))
            continue
        f = by_name.get(k)
        if f is None:
            valid = sorted({g.name for g in msgs} | {f"f{g.num}" for g in msgs} | {_codec.UNKNOWN})
            raise KeyError(f"{k!r} is not a field of {msg_type}; valid keys: {valid}")
        if f.num in seen:
            raise ValueError(f"field {f.num} of {msg_type} supplied twice: {seen[f.num]!r} and {k!r}")
        seen[f.num] = k
        els = [v] if f.packed else (v if isinstance(v, list) else [v])
        for el in els:
            items.append((f.num, f, el if f.map else _codec._unlab(f, el)))
    # preserve insertion order for underscore messages
    parts = []
    for fn, f, el in items:
        if f is None:
            wt, raw = el
            parts.append(_codec._enc_varint(fn << 3 | wt))
            if wt == 2:
                parts.append(_codec._enc_varint(len(raw)))
            parts.append(raw if isinstance(raw, bytes) else _codec._enc_varint(raw))
        else:
            parts.append(_encode_field_wrapper(fn, f, el))
    return b"".join(parts)

def _encode_field_wrapper(fn, f, v):
    if f.map:
        kt, vt = f.map
        inner = []
        if v[0] is not None:
            inner.append((1, kt, v[0]))
        if v[1] is not None:
            inner.extend((2, vt, e) for e in (v[1] if isinstance(v[1], list) else [v[1]]))
        blob = b"".join(_emit_wrapper(n, t, e) for n, t, e in inner)
        return _codec._enc_varint(fn << 3 | 2) + _codec._enc_varint(len(blob)) + blob
    if f.packed:
        runs = v if (v and isinstance(v[0], list)) else [v]
        return b"".join(
            _codec._enc_varint(fn << 3 | 2) + _codec._enc_varint(len(p)) + p
            for p in (b"".join(_codec._enc_scalar(f.type, e) for e in run) for run in runs))
    return _emit_wrapper(fn, f.type, v)

def _emit_wrapper(num, ftype, v):
    if ftype.startswith("M:"):
        blob = encode(v, ftype[2:])
        return _codec._enc_varint(num << 3 | 2) + _codec._enc_varint(len(blob)) + blob
    return _codec._enc_varint(num << 3 | _codec._wt_of(ftype)) + _codec._enc_scalar(ftype, v)

def encode(obj, msg_type):
    if isinstance(msg_type, str) and msg_type.startswith("_"):
        return _encode_unsorted(obj, msg_type)
    return _orig_encode(obj, msg_type)

def decode(data, msg_type):
    return _orig_decode(data, msg_type)

_codec.encode = encode
_codec.decode = decode
