import type { Draft, Kind } from './contract'
export const examples: { name: string; draft: Draft }[] = [
  {
    name: 'Play music',
    draft: {
      kind: 'choice',
      format: 'text',
      state: 'Can you play some jazz for dinner?',
      instructions: 'Choose the intent that best matches this request.',
      options: [
        { id: 'play_music', description: 'The user wants to play music or a specific genre.' },
        { id: 'book_restaurant', description: 'The user wants to find or book a restaurant.' },
        { id: 'get_weather', description: 'The user wants to check the weather forecast.' },
      ],
    },
  },
  {
    name: 'Find a restaurant',
    draft: {
      kind: 'choice',
      format: 'text',
      state: 'Book a table for two at an Italian restaurant tonight.',
      instructions: 'Choose the intent that best matches this request.',
      options: [
        { id: 'play_music', description: 'Play music.' },
        { id: 'book_restaurant', description: 'Find or book a restaurant.' },
        { id: 'get_weather', description: 'Check the weather.' },
      ],
    },
  },
  {
    name: 'Check the weather',
    draft: {
      kind: 'choice',
      format: 'text',
      state: 'Will it rain in Seattle tomorrow?',
      instructions: 'Choose the intent that best matches this request.',
      options: [
        { id: 'play_music', description: 'Play music.' },
        { id: 'book_restaurant', description: 'Find or book a restaurant.' },
        { id: 'get_weather', description: 'Check the weather.' },
      ],
    },
  },
  {
    name: 'Classify support intent',
    draft: {
      kind: 'choice',
      format: 'json',
      state: JSON.stringify(
        {
          message: 'I was charged twice for my subscription this month.',
          account: { plan: 'monthly', status: 'active' },
        },
        null,
        2,
      ),
      instructions: 'Route this support request to the most relevant team.',
      options: [
        { id: 'billing', description: 'Payments, charges and refunds.' },
        { id: 'technical', description: 'Bugs, outages and account access.' },
        { id: 'sales', description: 'Plans, pricing and upgrades.' },
      ],
    },
  },
  {
    name: 'Rate a reply',
    draft: {
      kind: 'score',
      format: 'text',
      state:
        'Customer: I was charged twice.\nReply: I’m sorry about the duplicate charge. I’ll check both transactions and arrange a refund if the second charge is a duplicate.',
      instructions:
        'Rate how helpful the reply is, considering relevance, clarity and an actionable next step. Use the ordered levels.',
      options: [
        { id: '0', description: 'Not helpful: ignores the request.' },
        { id: '1', description: 'Slightly helpful: acknowledges it but offers no next step.' },
        { id: '2', description: 'Moderately helpful: relevant but incomplete.' },
        { id: '3', description: 'Helpful: relevant, clear and actionable.' },
        { id: '4', description: 'Very helpful: complete, specific and empathetic.' },
      ],
    },
  },
  {
    name: 'Is this spam?',
    draft: {
      kind: 'noul',
      format: 'text',
      state:
        'Congratulations! You have won a free holiday. Click this unfamiliar link now and enter your bank details to claim it.',
      instructions: 'Does this message look like spam?',
      options: [
        { id: 'false', description: 'An ordinary message with no spam signals.' },
        { id: 'true', description: 'An unsolicited or deceptive promotional message.' },
      ],
    },
  },
]
export function defaultDraft(kind: Kind): Draft {
  return structuredClone(examples.find((e) => e.draft.kind === kind)!.draft)
}
