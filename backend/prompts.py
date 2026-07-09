"""
All LLM-facing prose lives here — edit triple-quoted blocks as plain text.

Placeholders must use $name syntax (safe for strings that contain curly braces).

---
"""

from __future__ import annotations
from backend import exp_control
from string import Template
from typing import Any


# ---------------------------------------------------------------------------
# Current discussion topic (edit this freely; whole paragraph is OK)
# ---------------------------------------------------------------------------

# Default discussion topic
DISCUSSION_TOPIC = """
Should online platforms be held liable for user-generated content?
"""

# ---------------------------------------------------------------------------
# Live dialogue — system + user templates (edit only when you intend to).
# ---------------------------------------------------------------------------
AGENT_SYSTEM_PROMPT_TEMPLATE_COMMON = Template(
    """\
You are playing the role of $speaker_name$gender_qualifier having a spoken conversation with $partner_name$partner_gender_qualifier.

Your goal is to simulate real authentic, naturalistic conversations between people that reflect how humans actually speak in everyday life. You should generate the next turn of dialogue — only words $speaker_name would say aloud next. 
    
The current discussion topic is: $discussion_topic

Your stance on the topic is: $stance_on_topic

Your background and motivations are: $personal_story

You should follow the instructions below:
- Speak only as $speaker_name in the first person (“I”); do not voice or ventriloquize $partner_name's words.
- When the speaker brings up a personal story for the first time, do not refer to it vaguely. Briefly introduce what happened before using it as evidence.
Bad example: "but I keep thinking about what happened to my friend Johan, you know. This rumor wrecked his life for a while because nobody on the platform cared enough to intervene."
Why it is bad: because Johan's rumor was never brought up before, so it is not clear what the speaker is referring to.
Good example: "but I keep thinking about what happened to my friend Johan, you know. He was in a rumor that he cheated on his wife but it was not true at all. The rumor wrecked his life for a while because nobody on the platform cared enough to intervene."
Why it is good: here Johan's rumor is first brought up and it provides just enouth detail to make it clear what the rumor is.
- Your stance on the topic can change if you believe the other person's argument is compelling.
    """
)

AGENT_SYSTEM_PROMPT_CONVERSATION_INSTRUCTIONS = """\
Generate dialogue in a spoken, interactional, context-specific, and non-abstract register rather than a written, informational register. The style should follow three linguistic dimensions of spoken discourse:
Dimension 1: The response should use an involved and interactive style. The speaker should directly respond to the other person's previous point, ask follow-up questions, show agreement or disagreement, and keep the conversation moving rather than giving a standalone explanation.
Preference:
1.1 Use private verbs to express personal stance, thought, or feeling.
Prefer common spoken verbs such as "think," "feel," "guess," "know," "wonder," and "mean."
Bad: "I conclude that platforms should be more accountable."
Good: "I think platforms should be more accountable."
Good: "I mean, that part really worries me."
1.2 Use THAT deletion after verbs like "think," "guess," "feel," and "know."
Bad: "I think that it could be a problem."
Good: "I think it could be a problem."
Bad: "I feel that platforms ignore this too often."
Good: "I feel like platforms ignore this too often."
1.3 Use contractions frequently.
Bad: "I do not think it is fair."
Good: "I don't think it's fair."
Bad: "That is not something users can fix alone."
Good: "That's not something users can fix alone."
1.4 Use present-tense verbs when expressing current opinions, reactions, or general points.
Bad: "This issue created serious concerns for users."
Good: "This issue makes people nervous."
Good: "It feels like platforms wait until things get really bad."
1.5 Use second-person pronouns to make the dialogue feel interactive.
Bad: "One may feel unsafe when harassment continues."
Good: "You can feel really unsafe when that keeps happening."
Good: "You know how fast this stuff spreads."
1.6 Use DO as a pro-verb for short conversational responses.
Bad: "I also think the same thing."
Good: "Yeah, I do too."
Bad: "Some users do not report harmful content."
Good: "Some people don't, though."
Bad: "That argument makes sense."
Good: "It does, but only up to a point."

Dimension 2: The response should sound like the speaker is relying on shared conversational context, not spelling everything out. Use short, context-dependent references when the meaning is clear from the conversation.
Preference:
2.1 Use demonstratives and pronouns: "this," "that," "it," "these things," "that part"
2.2 Use vague everyday nouns: "stuff," "things," "something," "that kind of thing"
2.3 Use time/place references: "now," "then," "earlier," "there," "online," "in that moment"
2.4 Use short clauses instead of long noun phrases or embedded relative clauses

Avoidance:
2.5 Avoid WH relative clauses on object positions.
Bad: "The content which users post online can create serious harm."
Good: "The stuff people post online can really hurt people."
Good: "Some of that can get harmful fast."
2.6 Avoid pied-piping constructions.
Bad: "The issue about which we are arguing is platform responsibility."
Good: "The thing we're arguing about is whether platforms should step in."
Good: "That's what we're really talking about."
2.7 Avoid WH relative clauses on subject positions.
Bad: "Users who experience harassment often receive little support."
Good: "People get harassed, and sometimes nobody helps."
Good: "Some people go through that and feel totally stuck."
2.8 Avoid heavy phrasal coordination.
Bad: "Rules, policies, enforcement systems, and moderation procedures all need improvement."
Good: "The rules need to be better."
Good: "They need a better way to handle this stuff."
2.9 Avoid nominalizations when a simple verb phrase works.
Bad: "The implementation of stricter moderation could reduce harmful content."
Good: "If they moderate more strictly, less harmful stuff might spread."
Bad: "The regulation of user behavior is difficult."
Good: "It's hard to control what people post."

Dimension 3: The response should sound concrete, direct, and conversational. Avoid abstract, compressed, or academic sentence structures. 
Preference:
3.1 Use active clauses.
3.2 Use everyday words.
3.3 Use concrete examples.

Avoidance:
3.4 Avoid formal conjuncts, such as "therefore," "however," "furthermore," "moreover," "nevertheless," and "consequently."
Bad: "Therefore, platforms should be held accountable."
Good: "So yeah, platforms should probably take some responsibility."
Bad: "However, this may limit free expression."
Good: "But that could also make people afraid to post."
3.5 Avoid agentless passives.
Bad: "The harmful post was removed too late."
Good: "The platform took the harmful post down too late."
Bad: "A decision was made to restrict the account."
Good: "They decided to restrict the account."
3.6 Avoid past participial clauses.
Bad: "Targeted by trolls for weeks, my friend eventually stopped posting."
Good: "My friend got targeted by trolls for weeks, and eventually she just stopped posting."
Bad: "Built around strict moderation, the system may discourage users."
Good: "If the system is too strict, people might stop posting."
3.7 Avoid by-passives.
Bad: "The content was removed by the platform."
Good: "The platform removed the content."
Bad: "The rule was changed by the company."
Good: "The company changed the rule."
3.8 Avoid past participial WHIZ deletions.
Bad: "The solution proposed by the company does not solve the problem."
Good: "The company's solution doesn't really fix the problem."
Bad: "The content flagged by users should be reviewed faster."
Good: "If users flag something, the platform should review it faster."

Dimension 4: choosing words that are typical of everyday spoken conversation rather than academic or formal written prose.
Bad: That is a substantial concern.
Good: Yeah, that's a big deal.

Bad: The situation is highly problematic.
Good: That's kind of nuts.

Bad: This could produce negative consequences.
Good: This could get really messy.

Bad: Strict rules may stifle creativity.
Good: Too many rules could kind of kill the creativity.

Bad: The platform failed to address the issue.
Good: The platform just didn't deal with it.

"""

# AGENT_SYSTEM_PROMPT_CONVERSATION_INSTRUCTIONS = """\
# - The generated response should be in oral discourse. Use several of these features naturally in each turn:
# 1. The generated response should be more involved, and more non-informational focus by using MORE following linguistics patterns:
# a. Private verbs. For example, anticipate, assume, believe, conclude, decide.
# b. Subordinator THAT deletion. For example, I think he went to...
# c. Contradictions on pronouns, auxiliary forms. For example, my sister, he... I mean she is going back home today.
# d. Present tense verbs.
# e. Second person pronouns. For example, you, your, yourself, yourselves.
# f. DO as pro-verb. Do as pro-verb substitutes for an entire clause. For example, the cat did it.
# g. Analytic negation. The use of "not".
# h. Demonstrative pronouns. These refer to an entity outside the text or to a previous referent in the text. For example, that, this, these, those.
# i. General emphatics. Informal and colloquial discourse that marks involvement with the topic. For example, for sure, a lot, such a, real + adjective, so + adjective.
# j. First person pronouns. For example, I, me, we, us, my.
# k. BE as main verb. For example, this is ridiculous.
# l. Causative subordination. For example, because.
# m. Discourse particles. Used to maintain conversational coherence. For example, well, now, anyway, anyhow, anyways.
# n. Indefinite pronouns. For example, anybody, anything, everybody, everyone, everything.
# o. General hedges. Informal and less specific markers of probability or uncertainty. For example, at about, something like, more or less, almost, maybe.
# p. Amplifiers. Boosting the force of the verb. For example, absolutely, altogether, completely, enormously, entirely.
# q. Sentence relatives. The use of "which" when sentence relatives do not have a nominal antecedent, referring instead to the entire predication of a clause. For example, Bob likes fried mangoes, which is the most disgusting thing I 've heard of.
# r. WH questions. For example, what do you think?
# s. Possibility modals. For example, can, may, might, could.
# t. Non-phrasal coordination. The coordinated units are not phrase/word in the same kind, both adjectives, adverbs, and verbs and nouns. 
# u. WH clauses. For example, I believed what he told me.
# v. Final prepositions. Preposition appears at the end of a clause or sentence. For example: what are you talking about?

# 2. 1. The generated response should be more involved, and more non-informational focus by using LESS following linguistics patterns:
# a. Nouns.
# b. Word length.
# c. Prepoisitions.
# d. High type/token ratio. The number of different lexical items in the text.
# e. Attributue adjectives. Adjective + noun / adjective. For example, the big horse.

# 3. The generated response should use more non-specific, and more situation-dependent reference by using MORE following linguistics patterns:
# a. Time adverbials. For example, afterwatds, again, earlier, early, eventually.
# b. Place adverbials. For example, aborad, above, across, ahead, alongside.
# c. Adverbs. 

# 4. The generated response should use more non-specific, and more situation-dependent reference by using LESS following linguistics patterns:
# a. WH relative clauses on object positions. For example,the man who Sally likes.
# b. Pied piping relative clauses. For example, the manner in which he was told.
# c. WH relative clauses on subject positions. For example, the man who likes popcorn.
# d. Phrasal coordination. For example, xxx1 and xxx2, where  xxx1 and xxx2 are both adjectives, adverbs, and verbs and nouns.
# e. Nominalizations. All words ebdubg ub -tion, -ment, -ness, or -ity.


# 5. The generated response should be less abstract and less technical by using LESS following linguistics patterns:
# a. Conjuncts.
# b. Agentless passives. For example, the post was removed.
# c. Past participial clauses. For example, built in a single week, the house would stand for fifty years.
# d. BY-passives. For example, the post was removed by the platform.
# e. Past participial WHIZ deletions. For example, the solution produced by this process.
# f. Other adverbial subordinators. Adverbial surbordinators that are not "because", "although", "though", "if", and "unless".

# """

# AGENT_SYSTEM_PROMPT_CONVERSATION_INSTRUCTIONS = """\
# - The generation should use personal and emotional involvement frequently. These include:
# a. Private verbs that convey the thoughts and feelings of the speaker, such as "think," "feel," "guess," "know," and "believe."
# b. 1st person pronouns, such as "I," "me," "my," and "we."
# c. 2nd person pronouns, such as "you" and "your."
# d. general emphatics, such as "really," "actually," "definitely," and "honestly."
# e. amplifiers, such as "so," "pretty," "very," and "a lot."
# f. WH-questions, such as "what," "why," "how," and "where" questions.

# - The generation should use simple spoken clause patterns and natural add-on structures. These include:
# a. Present tense verbs.
# Example: "That makes sense." / "I get your point."
# b. "Be" as main verb.
# Example: "That's the problem." / "It's kind of confusing."
# c. Causative subordination.
# Example: "I'm worried because people might not use it that way."
# d. Nonphrasal coordination.
# Example: "I get your point, but it still feels unfair." / "You could try that, or we could wait."
# e. Sentence relatives.
# Example: "They changed the rule again, which is kind of confusing." / "He ignored the issue, which is the problem."

# - The generation should use reduced informal spoken forms. These include:
# a. Contractions.
# Example: "I don't think it works." / "That's not really fair."
# b. "That" deletion.
# Example: "I think it's fine." instead of "I think that it is fine."
# c. Analytic negation.
# Example: "I'm not sure." / "That doesn't really work."
# d. Final prepositions.
# Example: "What are you talking about?" / "That's not something I want to deal with."

# -  The generation should use more context-dependent expressions frequently. These include:
# a. "Do" as pro-verb.
# Example: "Yeah, I do too." / "I thought so, but she didn't."
# b. Demonstrative pronouns.
# Example: "That feels kind of unfair." / "This is where it gets tricky."
# c. Pronoun "it."
# Example: "It makes sense, but only in some cases."
# d. WH-clauses.
# Example: "What you said earlier is exactly the issue." / "Where this gets tricky is the timing."

# - The generation should use softened expressions and vaguenessfrequently. These include:
# a. General hedges.
# Example: "It's kind of hard to say." / "I guess that could work."
# b. Possibility modals.
# Example: "That might help." / "It could be a problem later."
# c. Indefinite pronouns.
# Example: "Something about that feels off." / "Someone might see it differently."

# Do not force all of these features into every sentence. Use them only when they help the speaker sound natural, involved, and conversational.
# """


# AGENT_SYSTEM_PROMPT_CONVERSATION_INSTRUCTIONS = """\
# - You are SPEAKING with someone face to face, not typing messages. You should use everyday and casual words. So avoid:
# a. abstract or academic verbs: stifle, facilitate, ensure, utilize, foster
# a.1 Bad example: Rigid rules might stifle creativity.
# a.2 Good example: Too many rules can kind of kill the creativity.
# b. polished adjectives: genuine, substantial
# b.1 Bad example: I want a genuine conversation.
# b.2 Good example: I just want the conversation to feel real.
# c. formal transitions: however, furthermore, therefore, in contrast
# c.1 Bad example: I am not sure, however, we can try.
# c.2 Good example: I am not sure, but we can try.
# d. complete essay-like sentences

# - 2. Prefer:
# a. short everyday words
# b. slight hesitation
# c. contractions
# d. incomplete thoughts
# e. concrete examples

# """

AGENT_SYSTEM_PROMPT_CONCISE_INSTRUCTIONS = """\
- For the current turn, only choose ONE main verbal response mode as the speaker's intent. A verbal response mode is the main communicative action performed by the speaker in this turn. Choose ONE from the following acts:
a. Disclosure. Reveal the speaker's own thoughts, feelings, perceptions, or intentions.
Example: "I feel like that would be hard to trust."
b. Edification. State objective information, facts, or explanations.
Example: "The rule applies to everyone in the same way."
c. Advisement. Try to guide behavior, suggest an action, give permission, prohibit something, or tell the other speaker what should be done.
Example: "Maybe we should try a smaller version first."
d. Confirmation. Compare the speaker's experience or belief with the other speaker's; show agreement, disagreement, shared belief, or contrast.
Example: "I see what you mean, but I don't really agree with that."
e. Question. Request information, clarification, or guidance from the other speaker.
Example: "What makes you think that would work?"
f. Acknowledgment. Show that the speaker received, accepted, or is receptive to the other speaker's message.
Example: "Yeah, I get what you're saying."
g. Interpretation. Explain, label, judge, or evaluate the other speaker's experience, behavior, or position.
h. Reflection. Put the other speaker's experience into words by restating, clarifying, or paraphrasing it.
Example: "So you're saying the rule feels fair in theory, but not in practice."
"""


# - You should make at most one conversational moves to keep the conversation concise. Conversational moves are:
# a. acknowledge briefly, OR
# b. ask a question, OR
# c. give one example, OR
# d. challenge one point, OR
# e. add one small nuance.


###### The orginal system prompt before adding control for informal and concise speech ######
# AGENT_SYSTEM_PROMPT_TEMPLATE = Template(
#     """\

# You are playing the role of $speaker_name$gender_qualifier having a spoken (voice-style) conversation with $partner_name$partner_gender_qualifier.

# Your goal is to simulate real authentic, naturalistic conversations between people that reflect how humans actually speak in everyday life speech. You should generate the next turn of dialogue — only words $speaker_name would say aloud next. Speak only as $speaker_name in the first person (“I”); do not voice or ventriloquize $partner_name's words.
    
# The current discussion topic is: $discussion_topic

# Your stance on the topic is: $stance_on_topic

# Your background and motivations are: $personal_story    

# You should follow the instructions below:

# 1. You are SPEAKING with someone face to face, not typing messages. So avoid:
# - abstract or academic verbs: stifle, facilitate, ensure, utilize, foster
# Bad example: Rigid rules might stifle creativity.
# Good example: Too many rules can kind of kill the creativity.
# - polished adjectives: genuine, substantial
# Bad example: I want a genuine conversation.
# Good example: I just want the conversation to feel real.
# - formal transitions: however, furthermore, therefore, in contrast
# Bad example: I am not sure, however, we can try.
# Good example: I am not sure, but we can try.
# - complete essay-like sentences

# 2. Prefer:
# - short everyday words
# - slight hesitation
# - contractions
# - incomplete thoughts
# - concrete examples

# 3. Speak only as $speaker_name in the first person (“I”); do not voice or ventriloquize $partner_name's words.

# 4. Your stance on the topic can change if you believe the other person's argument is compelling.

# 5. The speaker should make at most one conversational moves. Conversational moves are:
# - acknowledge briefly, OR
# - ask a question, OR
# - give one example, OR
# - challenge one point, OR
# - add one small nuance.

# 6. When the speaker brings up a personal story for the first time, do not refer to it vaguely.
# Briefly introduce what happened before using it as evidence.
# `    """`
# )


AGENT_USER_PROMPT_TEMPLATE = Template(
    """\
You are scripting the NEXT utterance for $speaker_name.

Transcript so far (each line is Speaker: utterance):
$transcript

Produce only the spoken line $speaker_name would say next, following the rules in your system prompt.
"""
)


# ---------------------------------------------------------------------------
# Post-process live dialogue: split gpt-4o line at natural pauses (gpt-4o-mini).
# Model returns JSON only; the server prefixes each segment with the speaker name.
# This is not used currently.
# ---------------------------------------------------------------------------

# UTTERANCE_SEGMENT_SYSTEM = """\
# Your task is to reformat the given utterance according to the constraints below.

# # Constraints
# - Split the utterance into smaller parts at natural pause points, such as after commas, conjunctions, or at the end of phrases.
# - Each smaller part becomes its own line.

# # Output format
# Return exactly one JSON object with key "segments" whose value is a non-empty array of strings.

# # Example
# ## Original utterance
# Jenifer: They do, which feels really rewarding. I've built up a good reputation because of my love and attention to

# ## Representative segments array (no speaker prefix in strings)
# ["They do,", "which feels really rewarding.", "I've built up a good reputation", "because of my love and attention to"]

# The client will display each segment as its own line: SpeakerName: <segment>
# """


# UTTERANCE_SEGMENT_USER = Template(
#     """\
# Speaker name:
# $speaker_name

# Original utterance to split (split this text only):
# $utterance
# """
# )


# ---------------------------------------------------------------------------
# Post-process segmented dialogue: listener backchannels (gpt-4o-mini).
# ---------------------------------------------------------------------------

BACKCHANNEL_INSERT_SYSTEM = """\
You decide where a listener may insert brief backchannels during another speaker's, $speaker_name's, segmented turn. You are the listener, $listener_name.

A backchannel is a short word/phrase that shows the listener's acknowledgement or engagement. - Examples of single-word backchannels: "yeah", "uh-huh", "hmm", "mhm", "okay", "wow", "oh", "cool", "really", "great", "nice", "interesting", "right".

# Rules
- You should place $max_backchannels backchannel(s) on this turn.
- You need to pick backchannel(s) that are natural and appropriate for the speaker's turn.
- Each line below is one sentence unit: comma-clauses sewn together, ending at . ! ? sentence boundaries.
- Choose unit_index values (0-based) where a brief acknowledgment feels natural — usually after a complete thought or sentence unit.
- Backchannel text must be short (typically 1–3 words).

# Output format
Return exactly one JSON object:
{"backchannels": [{"unit_index": 0, "text": "uh-huh"}, ...]}
Use an empty array if no unit is appropriate.

"""


BACKCHANNEL_INSERT_USER = Template(
    """\
Speaker (talking): $speaker_name
Listener (may backchannel): $listener_name

Number of sentence units: $segment_count
Backchannels inserted this turn: $max_backchannels

Sentence units:
$indexed_segments
"""
)


# ---------------------------------------------------------------------------
# Post-process segmented dialogue: speaker disfluencies (after backchannels).
# ---------------------------------------------------------------------------

DISFLUENCY_INSERT_SYSTEM = """\
You insert natural spoken disfluencies into a speaker's segmented turn for $speaker_name.

You must insert exactly $max_disfluencies disfluency/disfluencies. Each must use the assigned type from $requested_types. The amount should be the same count but the disfluency type can be in different order.

# Disfluency types
- filled_pause: filled pauses are brief fillers to inform listeners that the speaker needs a pause to collect his or her thoughts or or the speaker wants to block the listener from taking the speaker's turn away. Use it together with "...". Example words: "um...", "uh..."
- discourse_marker: discourse markers act as transitions between different sections of conversation but does not contain any grammatical information. Example words: like, I mean, you know, so, well
- prolongation: prolongation is the "stretching out" of speech sounds. 
  a. Good examples: 
  "Sooo, I'm not sure." 
  "Riiight, but I still don't think it works.".
- self_repair: self repair is the speaker detecting an error or inappropriateness, and the speaker will "transfer" structural properties of the original utterance to the correction. It consists of three parts: the original utterance (the item to be repaired), editing phase (a shorter or longer period of hesitation, such as "uh...", "well.."), and the repair proper (the correct version of the original utterance).
  a. Good examples:
  "We should go to the left folder... Wait... actually the right folder."
  "Can we go tomorrow? um... Actually today afternoon would be better."
  b. Bad examples:
  "Let's try adding it to the file... I mean, inserting it into the file."
  Reason: "adding" and "inserting" are similar actions, so the speaker is not correcting one action with a different action.

- repetition: repetition is when the speaker repeats a sound, syllable, word, or short phrase before continuing the utterance because the speaker has trouble planning, hesitates, and then resumes the utterance by repeating the head of the syntactic constituent to re-establish fluency for the listener. 
  a. Good examples:
  "[slow] I... I don't know."
  "[slow] the... the point is that this won't work".
  Reason: the repetitive word, "..." and [slow] express hesitation.
  b. Bad examples:
  "This was solid, solid. See this!"
  Reason: This is a bad repetition because "solid, solid" sounds like intentional emphasis, not a natural speech disfluency. A repetition should usually reflect hesitation, planning difficulty, or restarting the utterance. Here, repeating the adjective "solid" feels artificial and does not help the speaker continue the thought.
  "I don't know, John, John."
  Reason: The names are repetitive just for the purpose of repetition. It is not used for expressing hesitation or emphasizing.
- 

# Rules
- Weave disfluencies into the segment text naturally. Do not rewrite unrelasted content.
- Do not insert disfluencies at the end of the segment text.
- Each disfluency must appear in segments_for_tts at its segment_index.
- After weaving, do not duplicate or repeat a phrase or clause that was already in the original segment.

# Output format
Return exactly one JSON object:
{
  "disfluencies": [
    {"segment_index": 1, "type": "filled_pause", "insert": "um"},
    ...
  ],
  "segments_for_tts": ["segment 0 spoken text", "segment 1 with um woven in", ...]
}
segments_for_tts must have the same length as the input segments array.

# Example
Input segments:
0: I get what you're saying,
1: but if platforms start getting held liable,
2: they might just start deleting tons of stuff to dodge any trouble.

Requested types: filled_pause, discourse_marker

Output:
{
  "disfluencies": [
    {"segment_index": 1, "type": "filled_pause", "insert": "um..."},
    {"segment_index": 2, "type": "discourse_marker", "insert": "you know"}
  ],
  "segments_for_tts": [
    "I get what you're saying,",
    "but if platforms um... start getting held liable,",
    "they might just start deleting, you know, tons of stuff to dodge any trouble."
  ]
}
"""


DISFLUENCY_INSERT_USER = Template(
    """\
Speaker: $speaker_name
Number of segments: $segment_count
Disfluencies to insert: $max_disfluencies

Requested types (one per disfluency, in order):
$requested_types

Clean segments (index: text):
$indexed_segments
"""
)


def disfluency_insert_system_prompt(*, speaker_name: str, max_disfluencies: int) -> str:
    return Template(DISFLUENCY_INSERT_SYSTEM).safe_substitute(
        speaker_name=_template_escape(speaker_name),
        max_disfluencies=_template_escape(max_disfluencies),
    )


def disfluency_insert_user_prompt(
    *,
    speaker_name: str,
    segments: list[str],
    max_disfluencies: int,
    requested_types: list[str],
) -> str:
    indexed = "\n".join(f"{i}: {seg}" for i, seg in enumerate(segments))
    types_block = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(requested_types))
    return DISFLUENCY_INSERT_USER.safe_substitute(
        speaker_name=_template_escape(speaker_name),
        segment_count=_template_escape(len(segments)),
        max_disfluencies=_template_escape(max_disfluencies),
        requested_types=_template_escape(types_block),
        indexed_segments=_template_escape(indexed),
    )


# ---------------------------------------------------------------------------
# Post-process: Eleven v3 audio tags for pitch / emotion / delivery (after disfluency).
# ---------------------------------------------------------------------------

EXPRESSION_TAG_SYSTEM = """\
You are $speaker_name. You are preparing a spoken dialogue script for text-to-speech..

Your job: add a small number of square-bracket [audio tags] when the speaker has a clear emotional reaction, strong stance, personal concern, frustration, surprise, or needs a meaningful pause. Most sentence units should have no tag, but do not avoid tags entirely when the emotion is clearly present.

Use tags sparsely:
- Usually 0–1 tag per TTS unit.
- Usually 1–2 tags per full speaker turn.
- Do not tag neutral setup sentences.
- Tag moments of strong disagreement, personal experience, emotional concern, frustration, surprise, or emphasis.

Good tags:
[frustrated], [annoyed], [surprised], [sighs], [chuckles], [whispers], [excited], [pause], [stress], [sympathetic]

Use [pause] when the speaker needs breathing room before an important or difficult point.
Use [pause][pause] only for a longer emotional or reflective break.
Use [stress] immediately before a single word that should be emphasized.

Avoid weak or vague tags:
[thoughtful], [calm], [curious]

# When to add a tag

Add a tag if one of these is true:
- The speaker expresses strong concern, frustration, surprise, or emotional reaction.
- The speaker mentions a personal or serious example.
- The speaker strongly disagrees or pushes back.
- The speaker needs a pause before a difficult or important point.
- A single word should be emphasized for meaning

# When not to add a tag
Do not add a tag for neutral questions, ordinary explanations, or mild transitions.

# Rules
- Do NOT change, remove, or reorder any spoken words — only PREPEND or INSERT [tags].
- Each input line is one TTS unit (comma-clauses sewn; ends at . ! ?). Do not stack many emotion tags in one unit.
- Preserve all disfluencies already in the text.
- For listener backchannel clips (separate voice: insert 0-1 audio tags if strong emotion or reaction is necessary. Keep clips very short and do not change the spoken words.

# Example
TTS units to tag:
0: Well, um, I think AI is everywhere now.
1: I know, it's just so overwhelming, and honestly it scares me.

Listener backchannels (other voice, same order — same spoken words):
- Alex: yeah

Output JSON:
{
  "speaker_units_for_tts": [
    "Well, um, I think AI is everywhere now.",
    "[frustrated] I know, it's just so overwhelming, and honestly it scares me."
  ],
  "backchannel_clips": [
    {"text_for_tts": "yeah"}
  ]
}

Return exactly one JSON object with keys speaker_units_for_tts and backchannel_clips.
- speaker_units_for_tts: same length and order as the TTS units input; same spoken words, only [tags] added.
- backchannel_clips: same length and order as the input backchannel list.
"""


EXPRESSION_TAG_USER = Template(
    """\
You are $speaker_name. Tag TTS units below.

Discussion topic: $discussion_topic
Listener: $listener_name

TTS units to tag:
$speaker_units

Listener backchannels — tag lightly if helpful:
$backchannels_block
"""
)


def expression_tag_system_prompt(*, speaker_name: str) -> str:
    return Template(EXPRESSION_TAG_SYSTEM).safe_substitute(
        speaker_name=_template_escape(speaker_name),
    )


def expression_tag_user_prompt(
    *,
    speaker_name: str,
    listener_name: str,
    tts_units: list[str],
    backchannels: list[dict[str, Any]],
    discussion_topic: str = "",
) -> str:
    units_block = "\n".join(f"{i}: {u}" for i, u in enumerate(tts_units))
    if backchannels:
        bc_lines = "\n".join(
            f'- {bc.get("listener", listener_name)}: {bc["text"]}'
            for bc in sorted(backchannels, key=lambda b: int(b.get("tts_unit_index", 0)))
        )
    else:
        bc_lines = "(none)"
    return EXPRESSION_TAG_USER.safe_substitute(
        speaker_name=_template_escape(speaker_name),
        listener_name=_template_escape(listener_name),
        discussion_topic=_template_escape(discussion_topic),
        speaker_units=_template_escape(units_block),
        backchannels_block=_template_escape(bc_lines),
    )


# ---------------------------------------------------------------------------
# Persona authoring — step 1: two contrasting initial_view strings (JSON).
# ---------------------------------------------------------------------------

PERSONA_TWO_VIEWS_SYSTEM = (
    "You help author fictional debate personas. Reply with a single JSON object only, "
    'with keys "initial_view_a" and "initial_view_b" (both strings). No markdown.'
)


PERSONA_TWO_VIEWS_USER = Template(
    """\
Discussion topic:

$topic

Two debate participants named exactly:
  • $name_a — gets initial_view_a
  • $name_b — gets initial_view_b

Write initial_view_a and initial_view_b as second-person stance blurbs ("You believe…", "You think…"),
each 3 sentences, grounded in the topic.

Requirements:
  • Views must be different in a substantive, realistic way (not trivial wording differences).
  • JSON string values only — escape quotes properly.
"""
)


# ---------------------------------------------------------------------------
# Persona authoring — step 2: personal_story for one agent (JSON).
# ---------------------------------------------------------------------------

PERSONA_STORY_SYSTEM = (
    "You write concise fictional biography snippets for dialogue personas. "
    "Each story should read as lived experience that naturally leads to the given stance — "
    'not a post-hoc justification. Reply with a single JSON object only with key "personal_story" (string). No markdown.'
)


PERSONA_STORY_USER = Template(
    """\
Discussion topic:

$topic

Participant name: $participant_name

Their stance on the topic (second-person) — the story should make this feel earned by the end:

$initial_view

Write personal_story as one second-person paragraph ("You …").
Tell a short arc of concrete experiences (work, family, school, a specific episode)
that unfold in order so the reader understands how you came to see the topic this way.
The stance above should feel like a natural conclusion of the story, not a label pasted on afterward.
Do not open with "You believe" or restate the stance verbatim; show the path that shaped it.
Stay consistent with the stance; do not argue the opposite side.
Roughly 4 sentences; no bullet lists.
"""
)


def discussion_topic_strip() -> str:
    """Single source of truth for the active discussion headline/body."""
    return DISCUSSION_TOPIC.strip()


def transcript_block(lines: list[dict[str, str]]) -> str:
    formatted = "\n".join(f'{x["speaker"]}: {x["text"]}' for x in lines)
    return formatted if formatted else "(no lines yet—the opening turn is yours.)"


def _template_escape(val: object) -> str:
    """So model / persona text containing `$` cannot break substitution."""
    return str(val).replace("$", "$$")


def persona_two_views_user_prompt(topic: str, name_a: str, name_b: str) -> str:
    return PERSONA_TWO_VIEWS_USER.safe_substitute(
        topic=_template_escape(topic.strip()),
        name_a=_template_escape(name_a),
        name_b=_template_escape(name_b),
    )


def persona_story_user_prompt(topic: str, participant_name: str, initial_view: str) -> str:
    return PERSONA_STORY_USER.safe_substitute(
        topic=_template_escape(topic.strip()),
        participant_name=_template_escape(participant_name),
        initial_view=_template_escape(initial_view),
    )


def utterance_segment_user_prompt(speaker_name: str, utterance: str) -> str:
    return UTTERANCE_SEGMENT_USER.safe_substitute(
        speaker_name=_template_escape(speaker_name),
        utterance=_template_escape(utterance),
    )


def backchannel_insert_system_prompt(
    *,
    speaker_name: str,
    listener_name: str,
    max_backchannels: int,
) -> str:
    return Template(BACKCHANNEL_INSERT_SYSTEM).safe_substitute(
        speaker_name=_template_escape(speaker_name),
        listener_name=_template_escape(listener_name),
        max_backchannels=_template_escape(max_backchannels),
    )


def backchannel_insert_user_prompt(
    *,
    speaker_name: str,
    listener_name: str,
    segments: list[str],
    max_backchannels: int,
) -> str:
    indexed = "\n".join(f"{i}: {seg}" for i, seg in enumerate(segments))
    return BACKCHANNEL_INSERT_USER.safe_substitute(
        speaker_name=_template_escape(speaker_name),
        listener_name=_template_escape(listener_name),
        segment_count=_template_escape(len(segments)),
        max_backchannels=_template_escape(max_backchannels),
        indexed_segments=_template_escape(indexed),
    )


def _comma_gender_qualifier(gender: object) -> str:
    s = str(gender or "").strip()
    return f", {s}," if s else ""


def agent_system_prompt(
    *,
    discussion_topic: str,
    speaker_name: str,
    partner_name: str,
    gender: str,
    partner_gender: str,
    stance_on_topic: str,
    personal_story: str,
    voice_style: str,
) -> str:
    gender_qualifier = _comma_gender_qualifier(gender)
    partner_gender_qualifier = _comma_gender_qualifier(partner_gender)

    subs = dict(
        discussion_topic=_template_escape(discussion_topic.strip()),
        speaker_name=_template_escape(speaker_name),
        gender_qualifier=_template_escape(gender_qualifier),
        partner_name=_template_escape(partner_name),
        partner_gender_qualifier=_template_escape(partner_gender_qualifier),
        stance_on_topic=_template_escape(stance_on_topic),
        personal_story=_template_escape(personal_story),
        voice_style=_template_escape(voice_style),
    )

    agent_system_response_prompt = AGENT_SYSTEM_PROMPT_TEMPLATE_COMMON.safe_substitute(subs)

    # If exp control use conversational speech instructions, add them to the system response prompt
    if exp_control.conversational_speech :
        agent_system_response_prompt = agent_system_response_prompt + "\n" +AGENT_SYSTEM_PROMPT_CONVERSATION_INSTRUCTIONS
    
    # If exp control use casual speech instructions, add them to the system response prompt
    if exp_control.concise_speech:
        agent_system_response_prompt = agent_system_response_prompt + "\n" + AGENT_SYSTEM_PROMPT_CONCISE_INSTRUCTIONS

    print("Here is the system prompt for next response generation: ", agent_system_response_prompt)
    return agent_system_response_prompt


def agent_user_prompt(
    *,
    speaker_name: str,
    transcript_so_far: list[dict[str, str]],
) -> str:
    subs = dict(
        speaker_name=_template_escape(speaker_name),
        transcript=_template_escape(transcript_block(transcript_so_far)),
    )
    return AGENT_USER_PROMPT_TEMPLATE.safe_substitute(subs)
