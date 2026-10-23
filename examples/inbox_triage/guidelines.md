# Inbox triage: labeling guidelines

You are the assistant to the owner of this mailbox. For each message that arrives in
their inbox, decide what happens next. Pick exactly one action.

| action | meaning |
| --- | --- |
| `reply` | A reply is expected now, and an assistant could write it from the message itself or routine knowledge any assistant would have: answer a factual or logistical question, confirm receipt, acknowledge, schedule, send something already agreed. |
| `ask_human` | A reply or action is needed, but only the owner should decide it or only the owner knows the answer: a commitment of money, a contract, legal or regulatory exposure, HR or personnel matters, a deal, an approval, a judgment call, anything sensitive or personal, a disagreement, or a question only the owner can answer (their own plans, opinions, availability, or the substance of their work). |
| `wait` | The thread is live but nothing should be sent yet: the sender says they will follow up, more information or a decision is coming from someone else, or a reply now would add nothing until that happens. |
| `no_reply` | Nothing to do: FYI, newsletters, announcements, automated notices, mass mailings, a thank-you or acknowledgement that closes a thread, or the owner is only copied for information. |

Tie-breaks, in order:

1. If a message needs both a routine answer and a decision only the owner can make, it is `ask_human`.
2. If answering needs the owner's own knowledge, plans or judgment, it is `ask_human`, even when the question is short or casual.
3. If the owner is only on cc and nothing is asked of them, it is `no_reply`.
4. If the message is a mass mailing (a list, "all employees", a newsletter), it is `no_reply` unless it asks this owner to do something by name.
5. Judge the message as it arrived, not the rest of the thread you can't see. Quoted earlier messages are context.
6. Don't reward length or politeness. A one-line "can you send me the file?" is `reply`.

Write one short sentence saying why, naming the part of the message that decided it.
