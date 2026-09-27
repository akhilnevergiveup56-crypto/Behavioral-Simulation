# Behavioral Simulation Dataset

## Annotation Guidelines — Version 0.1

### 1. Purpose

The purpose of this dataset is to model plausible behavioral responses to complex situations given a person's stated emotional state, behavioral tendencies, cognitive tendencies, and optional personal/contextual information.

Annotators are estimating what the described person is likely to do. Annotators are not deciding what the person should do.

### 2. Core principle

Always evaluate:

**Person profile + relevant background context + specific scenario → plausible behavioral response**

Never derive a behavioral label from a single emotion or personality feature alone.

### 3. Behavioral targets

Rate each target from 0 to 100.

**Direct confrontation:** Directly challenge or address the other party, particularly in the immediate situation.

**Private discussion:** Address the issue later or in a private/one-to-one conversation.

**Withdrawal:** Reduce engagement, leave, disengage, become distant, or temporarily avoid the situation/person.

**Cooperative resolution:** Attempt to understand the situation and work collaboratively toward resolving the underlying problem.

**Delayed response:** Intentionally postpone action while processing emotions, gathering information, or considering consequences.

**Support seeking:** Seek advice, reassurance, emotional support, or practical help from another person.

Scores are independent. They do not need to sum to 100.

### 4. Score interpretation

0–20: Very unlikely

21–40: Unlikely

41–60: Plausible / uncertain

61–80: Likely

81–100: Very likely

Use the full range when appropriate. Avoid automatically assigning round values such as 50, 75, or 100 without considering the specific case.

### 5. Consider interactions

Consider the complete profile.

For example, high anger may increase confrontation tendency, but high self-control, high deliberative thinking, and high consequence awareness may reduce immediate confrontation.

Do not use simplistic rules such as:

"High anger = confrontation."

### 6. Scenario relevance

The same person may react differently to different situations.

Pay attention to:

* relationship between people
* perceived threat or unfairness
* social setting
* consequences
* power differences
* urgency
* available alternatives
* relationship importance
* uncertainty
* moral or practical stakes

Use only information actually present in the scenario or profile.

### 7. Personal/contextual information

Optional personal context may include family environment, relationship history, social environment, previous experiences, and self-reported life events.

Use contextual information only when it is relevant to the scenario.

Do not infer diagnoses, hidden trauma, abuse, personality disorders, or other clinical conditions from narrative descriptions.

Distinguish explicitly reported experiences from interpretations.

### 8. Behavioral rationale

After assigning scores, identify the 2–4 strongest factors influencing the prediction.

The rationale should explain how the profile interacts with the scenario.

Do not provide advice.

Do not describe what the person morally ought to do.

### 9. Natural-language response

Write approximately 150–300 words describing the most plausible response.

The response should:

* directly answer what the person would likely do
* explain why
* remain consistent with the profile
* reference relevant scenario details
* acknowledge meaningful internal conflict when appropriate
* avoid certainty

Prefer:

"The person would most likely..."

over:

"The person will definitely..."

### 10. Insufficient information

If the scenario or profile does not contain enough information for a meaningful assessment, mark the example as low-confidence or insufficient-information rather than inventing details.

### 11. Annotator confidence

After completing the annotation, rate confidence from 1 to 5:

1 — Very low
2 — Low
3 — Moderate
4 — High
5 — Very high

### 12. Example

Profile:

* Anger: 85
* Patience: 20
* Empathy: 75
* Impulsiveness: 80
* Self-control: 30
* Deliberative thinking: 35

Scenario:

"A close friend repeatedly cancels plans at the last moment. The person rearranged their schedule for an important event, but the friend cancelled again and dismissed the person's frustration."

Possible annotation:

* Direct confrontation: 72
* Private discussion: 81
* Withdrawal: 43
* Cooperative resolution: 64
* Delayed response: 38
* Support seeking: 21

Rationale:

"High anger, frustration, impulsiveness, and low patience increase the likelihood of addressing the issue rather than ignoring it. High empathy and the importance of the friendship make a private and relationship-oriented conversation more plausible than an openly hostile confrontation."

The numerical values above are illustrative and are not presented as objectively correct labels.