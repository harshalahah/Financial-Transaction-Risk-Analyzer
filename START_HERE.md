# Start Here — Plain English Guide & 10-Day Plan

This file is for you, not for the interviewer. `DOCUMENTATION.md` is the
technical reference; this is the version that makes it make sense first.

---

## 1. What this project is, in plain words

Imagine a bank where millions of payments happen every day. A few of them are
crimes: someone's stolen card being used, someone breaking a large pile of cash
into small deposits to hide it, someone letting a stranger's money pass through
their account for a fee.

A bank cannot have humans look at every payment. It can afford to look at maybe
**one in a hundred**. So the real question is never "can you spot fraud" — it's
**"if we can only look at one in a hundred payments, which hundredth do we
look at, and can we explain why?"**

This project answers that. Three parts:

**Part 1 — The data.** Real bank data is confidential, so I wrote a program
that makes fake but realistic transactions: 70,000 payments across 1,200
customers in rupees, over UPI, cards, IMPS, ATM and branch cash. Inside that
pile I deliberately hid three real crime patterns. I also hid *innocent things
that look guilty* — a family buying gold for a wedding, someone depositing
genuine cash, someone travelling abroad and spending in Dubai. That second bit
matters: without it, the computer could "catch" the crimes with a silly rule
like "big + foreign = fraud," and the results would be fake-good.

**Part 2 — Teaching the computer what "normal" looks like.** The key idea of
the whole project is this sentence:

> ₹40,000 is not suspicious. ₹40,000 is suspicious **for a person who has never
> spent more than ₹2,000.**

So instead of feeding the model raw amounts, I compute 25 numbers that describe
the payment **relative to that customer's own history**: how many times bigger
than their usual spend, how many payments they made in the last hour, whether
this phone is one they've used before, whether this country is new for them,
how close the amount is to ₹50,000 (the level above which a cash deposit
requires PAN — so people hiding money stop just below it). Those 25 numbers are
what the model actually learns from, and the most important one turned out to
be exactly that customer-relative ratio.

**Part 3 — Scoring and explaining.** A trained model gives each payment a risk
score from 0 to 1. Alongside it runs a set of 7 plain rules ("four online
payments within one hour", "third cash deposit just under ₹50,000 this week").
The final score is whichever is higher, so a rule can raise an alarm the model
missed but never silence one. Every alert comes back with the rules that fired
and the features that pushed the score up, because a bank has to be able to
tell a customer and a regulator *why* a payment was stopped. All of this is
wrapped in a FastAPI web service so another system can send a payment and get
a decision back in about 20 milliseconds.

**The result:** it catches roughly 95 out of every 100 suspicious payments
while flagging less than 1% of all payments for review.

### The three crime patterns, in one line each

- **Card fraud burst** — a thief tests a stolen card with a tiny ₹12 charge,
  then makes five big online purchases within the hour from a new phone.
- **Structuring (smurfing)** — instead of depositing ₹2 lakh in cash once
  (which requires PAN and leaves a trail), deposit ₹47,000 five times across a
  week.
- **Mule account** — a large transfer lands in an account and is pushed straight
  out again in a few chunks within hours, usually to people never paid before.

---

## 2. Is this too complex for 10 days?

**Understanding and explaining it: yes, that is very doable.** Nothing here is
exotic. It's pandas grouping, a scikit-learn pipeline, and a web framework whose
entire job is turning a Python function into an HTTP endpoint. There are maybe
eight ideas total, and you already understood the core one in section 1.

**Becoming genuinely fluent in FastAPI, scikit-learn and pandas in 10 days
while also preparing everything else: no.** That's not how skill works, and
you shouldn't pretend otherwise in the room. Nobody expects an intern to be
fluent. They expect you to understand what *you* built and be honest about
edges.

So the goal for 10 days is a specific, achievable one:

> Be able to explain every design decision in this project and defend it, run
> it live, modify one thing on request, and say clearly what you don't know yet.

An interviewer who asks "what would happen if I removed the `.shift(1)` on
line 68" and hears *"honestly I'd have to test it — my understanding is it would
leak the current transaction into its own baseline, which is why there's a test
asserting it doesn't"* thinks better of you than one who hears a bluffed answer.

### The one real risk

If you can't run the project and can't explain a line you claim to have
written, that's worse than not having the project. Every hour below is spent
on removing that risk.

### Day-by-day (≈1.5–2 hours a day, leaving room for everything else)

| Day | Focus | Done when you can… |
|---|---|---|
| 1 | Run it. `make install`, `make data`, `make train`, `make demo`. Read this file again. | …run the whole thing end to end and describe what each command produced |
| 2 | `generate_data.py`. Open `data/transactions.csv`, look at real rows. Change one thing (make structuring deposits ₹9,000 instead of ₹47,000) and rerun. | …explain the three typologies and why confounders were added |
| 3 | `features.py`, batch half only. Take one customer, work out 3 features by hand on paper. | …explain 5 features and why they're customer-relative |
| 4 | Leakage. Understand `rolling(...) - 1` and `.shift(1)`. Delete the shift, retrain, watch the score jump, put it back. | …explain leakage with your own example |
| 5 | `train.py`. Imbalance, PR-AUC vs accuracy, the time-based split, why the alert budget picks the model. | …explain why 99.6% accuracy would be worthless here |
| 6 | Threshold + metrics. Open `artifacts/metrics.json`, read every number, connect it to the confusion matrix. | …explain the precision/recall trade-off with the actual numbers |
| 7 | `service.py` + `api.py`. Start the server, open `/docs`, click through every endpoint. | …explain why rules sit next to the model, and what an endpoint is |
| 8 | Explanations + tests. Run `pytest -v`, read each test name, break one deliberately and watch it fail. | …explain the ablation explanation and the two tests that matter |
| 9 | Rehearse out loud. Say the 2-minute summary in §7.3 of DOCUMENTATION.md from memory, three times. Answer the Q&A in §7.2 without reading. | …do it without notes |
| 10 | Weak-spot patch. List the 5 questions that scare you most, write honest answers, practise saying "I don't know, but here's how I'd find out". | …feel bored of the project |

Day 4 and Day 9 are the two you must not skip.

### Things you should be able to define on Day 10

Feature engineering · class imbalance · precision vs recall · data leakage ·
train/test split (and why time-based) · overfitting · API endpoint · JSON
request/response · why a threshold isn't 0.5.

That's nine terms. That's the whole vocabulary you need.

---

## 3. Resume bullets

A word on your actual question first, because it matters more than the wording.
You asked for bullets that discourage deep technical questions. Vague bullets
don't do that — they do the opposite. "Worked on an AI system for fraud
detection" invites *"so what did you actually do?"*, which is the worst question
to face. **Specific bullets control the conversation**, because interviewers
follow the concrete hooks you give them. So the strategy isn't hiding; it's
putting three specific hooks on the page that you have prepared deep answers
for, so the questioning goes where you're strong.

Only claim what's true. If you regenerate the data or change a number, update
these.

**Recommended version (3 bullets, hooks you'll have rehearsed):**

> **AI-Powered Financial Transaction Risk Analyzer** | Python, FastAPI,
> scikit-learn, pandas
> - Built an end-to-end transaction monitoring system over ~70K synthetic
>   retail banking transactions, detecting card fraud, structuring and mule
>   account patterns at a 0.4% base rate.
> - Engineered 25 customer-relative behavioural features (spend deviation,
>   velocity windows, device and geography novelty) computed identically for
>   training and live inference, with tests guarding against data leakage.
> - Selected and tuned the model against a 1% analyst alert budget rather than
>   raw accuracy, reaching 95% recall; served scores through a FastAPI endpoint
>   that returns a plain-language reason for every alert.

The three hooks are *customer-relative features*, *alert budget*, and
*explainability* — the three things you'll have prepared best, and the three a
bank cares about most.

**Shorter version (2 bullets)** if space is tight:

> - Built a transaction monitoring system (Python, scikit-learn, FastAPI) over
>   70K synthetic banking transactions, flagging card fraud, structuring and
>   mule patterns with 25 customer-relative behavioural features.
> - Tuned detection against a realistic 1% review budget instead of raw
>   accuracy, reaching 95% recall, and served every score with a
>   plain-language explanation through a REST API.

**Two words to avoid:** don't write "deployed" (it isn't), and don't write
"real transaction data" (it isn't). Say **synthetic** on the resume itself —
it costs you nothing, and it means you can never be caught out. Stating it
first is a point in your favour; being asked about it is neutral; being caught
hiding it is fatal.

---

## 4. About the data feeling AI-generated

Two separate worries live inside this question, and they have different answers.

**"Is it obviously generic?"** — It was, and now it isn't. The dataset is now
Indian retail banking: amounts in INR with a median of ₹517, UPI making up 46%
of transactions, alongside CARD_POS, CARD_ONLINE, IMPS, NEFT, ATM and
BRANCH_CASH. Categories are kirana, fuel, food delivery, mobile recharge, gold
jewellery, P2P transfers. The structuring typology hides under the **₹50,000
PAN requirement** for cash deposits — a genuinely Indian regulatory detail that
a generic generator would never produce. The confounders are festival and gold
purchases. A reviewer reading `data/transactions.csv` sees a plausible Indian
bank ledger, not a textbook example.

**"Should I hide that the data is synthetic?"** — No, and you don't need to.
Synthetic data is the *correct* choice here and you can defend it in one
sentence: real labelled transaction data is confidential, no public dataset
carries AML typologies, and generating it let me control the base rate and the
difficulty. What would be damaging is claiming real data and being asked which
dataset. The documentation states it openly in section 1 and again in the
limitations — keep it that way.

### If you want to personalise it further (30 minutes, optional but worth it)

Small changes make it unmistakably yours and give you extra things to talk
about:

- In `generate_data.py`, swap the merchant categories for ones you actually
  use — Zomato, Swiggy, IRCTC, DMart, Jio recharge, auto/cab fare.
- Add a **salary-day effect**: a large credit on the 1st of each month for
  `account_type == "salary"` customers. Two lines, and it's a nice detail to
  mention.
- Add a **festival window** (say a two-week Diwali spike in October) where
  genuine spending rises across the board — a great confounder and a great
  thing to point at when asked about false positives.
- Change `RNG_SEED` to something personal so your dataset is not byte-identical
  to anyone else's.
- Commit the project to GitHub over several days with real commit messages,
  not one giant "initial commit". The history is part of the story.

Each of these takes minutes and each gives you a sentence that starts with
"one thing I added was…", which is the most convincing kind of sentence in an
interview.

---

## 5. If you only remember five things

1. The core idea: judge a payment against **that customer's own history**, not
   against a global rule.
2. Accuracy is useless at a 0.4% base rate; the right question is **how much
   crime you catch inside the alert budget you can staff**.
3. **Leakage** is using information you wouldn't have had at the time; every
   rolling window here excludes the current transaction, and a test proves it.
4. **Rules + model together** — rules are auditable, the model generalises, and
   rules can only raise risk, never suppress an alert.
5. Every alert carries a **reason**, because an unexplainable decision isn't
   deployable in a bank.
