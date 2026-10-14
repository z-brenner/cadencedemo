import type { Draft, Kind } from './contract'

// Fictional workflow examples. Cadence compares supplied answers; it gives no legal advice.
export const examples: { name: string; draft: Draft }[] = [
  {
    name: 'Route a document',
    draft: {
      kind: 'choice',
      format: 'text',
      state: 'Fictional document excerpt: Each party must keep the other party’s confidential information private and use it only to evaluate the proposed collaboration.',
      instructions: 'Choose the document category that best matches this excerpt.',
      options: [
        { id: 'confidentiality_agreement', description: 'An agreement about keeping shared information confidential.' },
        { id: 'services_agreement', description: 'An agreement describing services, deliverables and payment.' },
        { id: 'meeting_notes', description: 'Notes recording a meeting discussion and action items.' },
      ],
    },
  },
  {
    name: 'Route a review request',
    draft: {
      kind: 'choice',
      format: 'json',
      state: JSON.stringify({
        fictional: true,
        message: 'Please review this draft confidentiality agreement before we share the project plans.',
        attachment: 'draft_confidentiality_agreement.txt',
      }, null, 2),
      instructions: 'Route this fictional legal-work request to the most relevant work queue.',
      options: [
        { id: 'contract_review', description: 'Reviewing draft agreements and their clauses.' },
        { id: 'case_research', description: 'Researching authorities for a stated legal issue.' },
        { id: 'invoice_review', description: 'Checking legal invoices and billing entries.' },
      ],
    },
  },
  {
    name: 'Check a notice term',
    draft: {
      kind: 'noul',
      format: 'text',
      state: 'Fictional clause: Either party may end this agreement by giving the other party at least 30 days of written notice.',
      instructions: 'Does the provided clause explicitly require at least 30 days of written notice to end the agreement? Check only the supplied words.',
      options: [
        { id: 'false', description: 'No: the clause does not explicitly state that requirement.' },
        { id: 'true', description: 'Yes: the clause explicitly states at least 30 days of written notice.' },
      ],
    },
  },
  {
    name: 'Check automatic renewal',
    draft: {
      kind: 'noul',
      format: 'text',
      state: 'Fictional clause: This agreement lasts for one year. Any extension requires a new written agreement signed by both parties.',
      instructions: 'Does the provided clause explicitly say the agreement renews automatically? Check only the supplied words.',
      options: [
        { id: 'false', description: 'No: automatic renewal is absent, or an extension requires a new signed agreement.' },
        { id: 'true', description: 'Yes: the agreement explicitly renews automatically without a new signed agreement.' },
      ],
    },
  },
  {
    name: 'Prioritize review',
    draft: {
      kind: 'score',
      format: 'text',
      state: 'Fictional work item: A draft vendor agreement is waiting for review. The team’s internal target is tomorrow. A reviewer must check the payment terms before the team can finish its review.',
      instructions: 'Rate document-review priority using the supplied internal workflow levels. Use only the stated timing and work status; do not infer a legal deadline or legal risk.',
      options: [
        { id: '0', description: 'Archived: review is complete and no action remains.' },
        { id: '1', description: 'Low: review is needed but no internal target date is stated.' },
        { id: '2', description: 'Routine: review is needed with an internal target more than a week away.' },
        { id: '3', description: 'Soon: review is needed with an internal target in two to seven days.' },
        { id: '4', description: 'First: review is still needed and the internal target is today or tomorrow.' },
      ],
    },
  },
  {
    name: 'Prioritize an archive item',
    draft: {
      kind: 'score',
      format: 'text',
      state: 'Fictional work item: Review of the draft vendor agreement is complete. All review comments were addressed. The file is archived and no action remains.',
      instructions: 'Rate document-review priority using the supplied internal workflow levels. Use only the stated timing and work status; do not infer a legal deadline or legal risk.',
      options: [
        { id: '0', description: 'Archived: review is complete and no action remains.' },
        { id: '1', description: 'Low: review is needed but no internal target date is stated.' },
        { id: '2', description: 'Routine: review is needed with an internal target more than a week away.' },
        { id: '3', description: 'Soon: review is needed with an internal target in two to seven days.' },
        { id: '4', description: 'First: review is still needed and the internal target is today or tomorrow.' },
      ],
    },
  },
]
export function defaultDraft(kind: Kind): Draft {
  return structuredClone(examples.find((e) => e.draft.kind === kind)!.draft)
}
