---
layout: default
title: "MahjongBot: teaching a robot arm to play mahjong with my family"
description: "How I trained a low-cost SO-101 robot arm to play mahjong (mostly) by itself, so my family can play with it."
image: writeup/grandparents.jpg
next_url: /build-log/
next_title: "Build log: MahjongBot"
hero: true
eyebrow: "A robot arm · a family table · 2026"
pills: ["Autonomy: supervised", "Stage: internal test"]
hand: article_toc  # the contents strip down the left of the page: _data/article_toc.yml
---

## The short version
I’m Ethan, a product manager in Sydney. In five months, on a career break, I took a low-cost, open-source robot arm (SO-101) from nothing to winning full hands of mahjong unattended, and wrote the acceptance criteria that decide when it is safe to play with people. I'm building this with my family in mind. My grandparents, who play mahjong with us once a year when they visit from Malaysia. The goal is that a robot player can fill the empty seat the other 300+ days they're not physically with us. This is the first of several projects to learn robotics by doing and move my career into physical AI.


<!-- DRAFT (from review, section 2): optional short section to add right after "Why mahjong", about who it is for and what failure costs them:
## Who it’s for, and what failure costs them
The users are my grandparents, in their [age/80s?], who are not going to debug a robot. For them, a robot that stalls is mildly annoying, but a robot that quietly plays the wrong tile means the game they think they won, they didn’t. That asymmetry shaped every target below: fixes by a family member are cheap, silent errors are not, so I allow a few interventions per hand but zero unnoticed mistakes.
-->

**How it works, in brief:** The arm is steered by two kinds of software. A **vision model** looks through cameras to find and name tiles. A **policy**, a neural network trained on recorded examples of me doing the task, controls the arm’s movements to pick a tile up. Scripts handle the rest. Every new version of either one is a candidate I have to test before trusting it.

**Where it stands, 20 November 2026:** Robotto has won 2 full face-down hands cleanly, one of them unattended (17 and 19 November). A third attempt on 20 November stopped after 3 turns. It hasn’t played with a person yet. That starts once **8 of 10 consecutive face-down hands are clean: no silent errors, and no more than 3 interventions each**.

{% include youtube.html id="2sP834I__1Y" anchor="hero-video" caption="The robot playing a face-down hand to a win, unattended (19 November)" %}

{% include stats.html %}

## Why mahjong

My family only has enough mahjong “kaki” (buddies) to play when my grandparents are in town from Malaysia. That happens once a year.

{% include ig-empty-seat.html %}

{% include figure.html src="writeup/grandparents.jpg" caption="Grandparents playing mahjong with my sister and me. I always get beaten when they’re in town." %}

In June 2026 they were in Sydney. I’d just built two SO-101 arms: a follower that does the work and a leader I move by hand to teach it. Before they flew home, I wowed them with a classic pick-and-place demo of putting a ball into a container and let them remote-control it by hand (teleoperation).

I promised that one day “Robotto”, as they’d affectionately nicknamed it, would play mahjong with them. Whether they’re remote controlling it from back in Malaysia, or together with us in Sydney and wanting a break, Robotto will make sure we always have enough kaki on hand to play a game together. This write-up follows that promise. The first step is teaching Robotto to play mahjong by itself, before it learns to play with other people.

## The mahjong game loop

> Mahjong is a game typically played with four people. Each player holds **13 tiles**. On each **turn** you **draw a tile** from the **wall** and **throw one away**, aiming to be the first to turn your 14 tiles into a winning set of groups (aka **hand**).

{% include ig-game-loop.html %}

Non-goals: no specific regional rule set, no opponents yet, simply chasing a winning hand.

## Defining what success looks like

> The goal is for Robotto to play 8 "clean" hands out of 10 consecutive hands (defined below).

Before Robotto is trusted to play against people, it must first play mahjong by itself ("autonomous" play). To decide when it’s ready, I defined targets and guardrails that show the setup is robust, accurate and consistent.

- **Silent misreads are unacceptable.** The robot must never act on a tile name it got wrong without noticing.
- **A clean full hand** is the headline target.

### A “clean full hand”, spelled out

**Definition.** A clean hand runs from the start of the game, registering the 13 starting tiles, to a win (or the wall running out). In it, **a person performs no more than 3 interventions to help the robot recover from failure**.

{% include ig-interventions.html %}

In an ideal world, the robot plays a hand and a person’s only job is to feed it new tiles to draw from (via the wall). Robotto doesn’t need to be *fully* autonomous, but it shouldn’t need me to stop it or keep rescuing it either.

In a home setting I relaxed the targets by working back from what failure actually costs the users: fixes are cheap for my family. “No more than 3 interventions” is a judgement call that still asks Robotto to meet a high quality bar.

<!-- DRAFT (from review, section 2): one paragraph on a trade-off decision, to add after the paragraph above or in Lesson 3:
The decision I’m least sure about was shipping policy v13 as the default. In its first paired test it went 27 clean of 28 against 25 of 28 for the previous version, but with that few trials the statistics said "no difference shown". I shipped it anyway, because it picked all 28 tiles against 26, and because the alternative was weeks of extra recording to settle a question the next real hand would answer faster. I recorded that decision, the evidence and the caveat where I made it, so I could reverse it cheaply.
-->


### Dissecting a turn, step by step

Robotto must complete six key steps every turn in the game to reach a winning hand:

{% include turn-steps.html %}

{% include ig-kinds.html %}

> The pattern I took from this is to use fixed rules where the world is predictable, and learning where it is not. Finding that boundary took a few attempts.

{% include youtube.html id="Ba6jArcnLAQ" caption="Video of how the robot plays a turn in the mahjong game" %}

---

## System architecture (how it works)

So how does Robotto actually figure out which steps to perform and how to execute them? To play the game, some hardware is required:

{% include ig-architecture.html %}

{% include figure.html src="writeup/workspace.jpg" narrow=true caption="The workspace setup where I toiled my days and nights" %}

Before each game, a calibration check confirms the overhead camera maps the table to 3D positions the arm can reach. I then lay out a 13-tile hand and a two-tile-high wall to draw new tiles from each turn.

{% include figure.html src="writeup/tile-layout.jpg" caption="How the tiles are laid out for a mahjong game" %}

---

## The build & technical challenges

A good chunk of my time went on collecting data to train the vision models and ACT policies. It took far more hours of teleoperation and image annotation than I expected before they performed consistently and accurately.

It wasn't a straight line getting to this point. Here’s a high-level timeline of what got me to this point, followed by some technical challenges I encountered. If you’re interested in how the months played out in more detail, you can check out this other post [here]({{ '/build-log/' | relative_url }}).

{% include timeline-wall.html %}

### Challenge 1: Deep-learning can be brittle

I hadn’t realized how brittle learned policies can be — they sometimes struggle to adapt and generalize to unseen environments or scenarios unlike what it was trained on.

When collecting data, I made sure to capture images across a wide range of surfaces to stress-test the vision model. I recorded tile pickups across different orientations, a variety of table positions, lighting conditions, and surfaces, while incorporating DAgger for teaching failure recovery to build a robust policy from a diverse dataset.

{% include figure.html src="writeup/green-paper-test.jpg" caption="Creating a test set for the tile recognition vision model on a surface (green paper) outside its training set" %}

I would still get surprised at how it would fail even after doing all the above. It took several learned policy iterations to surface an issue with the arm being able to pick up a tile all on its lonesome, because it’d only ever seen me pick up tiles with many other tiles around it in its dataset.

{% include ig-lone-tile.html %}

{% include figure.html src="writeup/upside-down-pickup.jpg" caption="Example of what a bamboozled robot arm can look like, inventing a new upside down tile pickup style" %}

### Challenge 2: Simulation vs. real life

I held out demonstration data from table spots the ACT policies hadn’t seen during training to check if predicted movements from a newly trained policy matched my actual demonstrations. They worked well as quick sanity checks — like spotting if the gripper closed too early above the tile — but they couldn’t predict real-world performance as physical errors compounded over time.

In fact, two policy versions tied on every offline test (scoring the policy against recorded examples on the PC, without a physical trial on the arm). Yet on the physical arm, one picked up the tile 6 out of 14 times against 11 out of 14 for the other. It happened again in November: three versions tied offline, then went 23, 25 and 27 clean out of 28 on the arm. Those arm samples are small too (the November gaps were within noise), so the lesson isn’t that the arm is a perfect judge. It’s that offline scores didn’t predict which policy would work. Now, offline tests serve only as a preliminary screen before paired physical trials on a standardized setup.

{% include offline-vs-arm.html %}

### Challenge 3: Safety guardrails

Unlike SaaS, bugs executed on a robot can physically harm you (and there were certainly some close calls during this project). Safeguards were put in place to protect against erratic, sudden movements of the robot arm. Especially when controlled by a learned model that produces non-deterministic outputs.

{% include ig-guardrails.html %}

---

## Key takeaways

### Lesson 1: Working with AI coding agents as a PM

I didn’t write the code. AI coding agents did (600+ commits, 1,300 automated tests since the late-October rebuild). My job was the PM’s job: specify what "done" means, set acceptance criteria and guardrails, and own the decisions, including which policy ships, which data counts as an exam, what "clean" means, and the rules checked before anything moves the arm.

What made it work: every hardware lesson written down as a rule the agents read before touching that code, tests that run in a simulator (software test that runs commands without the physical arm) before anything reaches the arm, and a merge queue that holds decisions for me.

{% include ig-merge-flow.html caption="This process is the product of me merging seven branches in one evening with no conflicts. Bugs still got through." %}

### Lesson 2: Play found what practice couldn’t

Early play tests revealed systemic flaws that isolated practice of individual capabilities missed. It sounds straightforward, but I caught myself putting off these tests thinking it wasn’t ready for an end-to-end test yet. Instead, I would record more and more demonstrations for the ACT policy, or annotate more and more photos for the vision model.

{% include ig-there-and-back.html %}

Testing the complete workflow early is crucial — component tests miss what only the full system reveals.

### Lesson 3: How I decided what “good enough” meant

When you’re in the thick of it, constantly recording new demonstrations and retraining models can feel like you’re constantly going in circles. Each fix reveals fresh edge cases that demand more training. Setting clear benchmark tests was crucial for measuring real progress and preventing old capabilities from breaking when adding new ones.

{% include ig-checks.html %}

The one failure I treat as unacceptable is a **silent error**: a wrong tile played with nobody noticing. An assist costs a minute. A silent error means the game I thought it won, it didn’t.

---

## What’s next

At time of writing, Robotto has demonstrated it can achieve clean hands, just not consistently enough to hit the 8 clean out of 10 goal. Below I list what I'm doing to close the gap, and also hint at future scope to tackle beyond the interim goal.

{% include ig-next.html %}

---

## Closing thoughts and links

Coming from SaaS, this gave me a real appreciation for what it takes to make even a simple robotics demo reliable. I’m looking for product roles in robotics. If you build robots or play mahjong, I’d love to hear from you.

<!-- DRAFT (from review, section 2): sharper closing ask:
I’m looking for product roles in robotics, especially robots that work alongside people at home or in care. [If tailored: name the company and one sentence on why, e.g. what about their product connects to a grandparent sitting down to play a game.] If you build robots or play mahjong, I’d love to hear from you.
-->

- **Repo**: [mahjong-lekiwi](https://github.com/EthanYew/mahjong-lekiwi)
- **LinkedIn:** [https://www.linkedin.com/in/ethan-yew/](https://www.linkedin.com/in/ethan-yew/)
