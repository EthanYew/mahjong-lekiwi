// A pi extension that enforces, in the harness, what piq's prompt can only ask a 12B model to do. Loaded by piq/piq with `-e`.
//
//   1. An exact repeat of an earlier tool call is blocked. 2026-11-23: one question made 46 identical `read scripts/bench.py` calls in a row
//      (64 tool calls, 300 s, 54k tokens) although the prompt says not to read a file again.
//   2. After MAX_CALLS calls per question, every further call is blocked with "answer now". The prompt says "after about 5".
//   3. A turn that ends with reasoning and no text, no tool call (a blank answer, about 40% of replies to one question on 2026-11-23) gets up to
//      MAX_NUDGES nudges to answer. Turning thinking off did not help; this does.
//
// Each block's reason goes back to the model as the tool result, so it can act on it. Counters reset for every user prompt.
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const MAX_CALLS = 10;
const MAX_NUDGES = 2;

export default function (pi: ExtensionAPI) {
  let seen = new Set<string>();
  let calls = 0;
  let nudges = 0;

  pi.on("before_agent_start", async (event: any) => {
    if (event.messageType === undefined || event.messageType === "user") {
      seen = new Set();
      calls = 0;
      nudges = 0;
    }
  });

  pi.on("tool_call", async (event: any) => {
    const key = `${event.toolName}:${JSON.stringify(event.input)}`;
    if (seen.has(key)) {
      return {
        block: true,
        reason: "You already made this exact call and have its result above. Do not repeat it. Answer now from what you have, or make a different call.",
      };
    }
    if (calls >= MAX_CALLS) {
      return {
        block: true,
        reason: `Tool call limit reached (${MAX_CALLS}). Answer now in plain text with what you have and cite the files and lines you read. If you did not find the thing asked about, say it does not exist and name the nearest real option.`,
      };
    }
    seen.add(key);
    calls += 1;
  });

  pi.on("agent_before_settle", async (event: any) => {
    if (nudges >= MAX_NUDGES || event.outcome !== "completed") return;
    const entries: any[] = event.entries ?? [];            // only the drafts proposed so far at this boundary, not the conversation
    const messages: any[] = event.context?.contextMessages ?? [];
    const last = messages[messages.length - 1];
    if (last?.role !== "assistant") return;
    const content: any[] = Array.isArray(last.content) ? last.content : [];
    const hasText = content.some((c) => c?.type === "text" && String(c.text ?? "").trim().length > 0);
    const hasCall = content.some((c) => c?.type === "toolCall");
    if (hasText || hasCall) return;
    nudges += 1;
    return {
      entries: [
        ...entries,
        {
          type: "custom_message",
          customType: "piq-nudge",
          content: "Your last turn ended without an answer or a tool call. Reply now: answer the question in plain text, citing file and line, or say the docs don't cover it.",
          display: false,
        },
      ],
      continue: true,
    };
  });
}
