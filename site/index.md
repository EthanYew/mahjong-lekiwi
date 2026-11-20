---
layout: default
title: "MahjongBot: teaching a robot arm to play mahjong with my family"
description: "How I trained a low-cost SO-101 robot arm to play mahjong (mostly) by itself, so my family can play with it."
image: writeup/grandparents.jpg
next_url: /build-log/
next_title: "Build log: MahjongBot"
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

## Why mahjong

My family only has enough mahjong “kaki” (buddies) to play when my grandparents are in town from Malaysia. That happens once a year.

{% include figure.html src="writeup/grandparents.jpg" caption="Grandparents playing mahjong with my sister and me. I always get beaten when they’re in town." %}

In June 2026 they were in Sydney. I’d just built two SO-101 arms: a follower that does the work and a leader I move by hand to teach it. Before they flew home, I wowed them with a classic pick-and-place demo of putting a ball into a container and let them remote-control it by hand (teleoperation).

I promised that one day “Robotto”, as they’d affectionately nicknamed it, would play mahjong with them. Whether they’re remote controlling it from back in Malaysia, or together with us in Sydney and wanting a break, Robotto will make sure we always have enough kaki on hand to play a game together. This write-up follows that promise. The first step is teaching Robotto to play mahjong by itself, before it learns to play with other people.

## The mahjong game loop

> Mahjong is a game typically played with four people. Each player holds **13 tiles**. On each **turn** you **draw a tile** from the **wall** and **throw one away**, aiming to be the first to turn your 14 tiles into a winning set of groups (aka **hand**).

{% include figure.html src="writeup/game-loop.png" alt="How a hand of mahjong works. The tiles: dots, bamboo and characters numbered 1 to 9, plus winds and dragons, four copies of each. The goal: four sets and a pair, 14 tiles, where a set is a run of three in a row in one suit or a triplet of the same tile. Every turn: draw a tile so the hand has 14; if it is a winning hand, win; if not, discard one tile back to 13 and take the next turn. Robotto’s version plays alone, reads face-down tiles by lifting each drawn tile to a camera, and does not yet claim other players’ discards, score, or play against people." %}

Non-goals: no specific regional rule set, no opponents yet, simply chasing a winning hand.

## Defining what success looks like

> The goal is for Robotto to play 8 "clean" hands out of 10 consecutive hands (defined below).

Before Robotto is trusted to play against people, it must first play mahjong by itself ("autonomous" play). To decide when it’s ready, I defined targets and guardrails that show the setup is robust, accurate and consistent.

- **Silent misreads are unacceptable.** The robot must never act on a tile name it got wrong without noticing.
- **A clean full hand** is the headline target.

### A “clean full hand”, spelled out

**Definition.** A clean hand runs from the start of the game, registering the 13 starting tiles, to a win (or the wall running out). In it, **a person performs no more than 3 interventions to help the robot recover from failure**.

Interventions include:
- **Assist**: I correct a misread tile label, or physically reposition a tile, so the game can continue.
- **Safety stop**: I press Esc to halt the arm and prevent physical damage.
- **Reset**: I reset the arm after a tile lands more than 25 mm from where it should.

What’s not included:
- **Designed step**: something I am meant to do, such as setting up the wall each turn. Not an intervention.
- **Confirmation asks**: I approve a tile read without changing it. Not an intervention, but I count them, because each one is a person watching and waiting.
- **Recovery**: the robot notices its own failure and retries. Not an intervention.

At the end of the hand I run a post-game **audit**: I check the robot’s hand and discards against the table for any misidentified tile. It needs **zero mismatches**.

In an ideal world, the robot plays a hand and a person’s only job is to feed it new tiles to draw from (via the wall). Robotto doesn’t need to be *fully* autonomous, but it shouldn’t need me to stop it or keep rescuing it either.

In a home setting I relaxed the targets by working back from what failure actually costs the users: fixes are cheap for my family. “No more than 3 interventions” is a judgement call that still asks Robotto to meet a high quality bar.

<!-- DRAFT (from review, section 2): one paragraph on a trade-off decision, to add after the paragraph above or in Lesson 3:
The decision I’m least sure about was shipping policy v13 as the default. In its first paired test it went 27 clean of 28 against 25 of 28 for the previous version, but with that few trials the statistics said "no difference shown". I shipped it anyway, because it picked all 28 tiles against 26, and because the alternative was weeks of extra recording to settle a question the next real hand would answer faster. I recorded that decision, the evidence and the caveat where I made it, so I could reverse it cheaply.
-->


### Dissecting a turn, step by step

Robotto must complete six key steps every turn in the game to reach a winning hand:

{% include figure.html src="writeup/turn-steps.png" alt="One turn, step by step: six steps, three kinds of work. 1 Find (learned): YOLO on the overhead camera spots each tile and depth turns its pixels into a 3D position; face-down tiles in hand are tracked by saved slots. 2 Get above (geometry): IK moves the hand over the tile and a correction learned from hundreds of measured hovers cleans up the error. 3 Grasp (learned): an ACT policy lowers the hand, grabs the tile and lifts it, watching through the wrist camera. 4 Read (learned): the arm shows a drawn tile to the camera on its wheeled base and a small classifier names it. 5 Decide (rules): game code picks the discard that brings the hand closest to winning. 6 Return (geometry and learned): the drawn tile fills the discard’s gap, aimed by a correction fitted from over 1,000 measured set-downs." %}

Each step in a turn is done by one of three kinds of work:

- **Geometry:** Inverse kinematics (IK) is the math that works out how each motor needs to turn to place the gripper (robot claw) at a specific spot in space — like figuring out how to move your shoulder and elbow to reach a target point with your fingers.
- **Rules:** ordinary scripted code. Deterministic, predictable output. Cheap to implement, but rigid. A scripted tile pickup motion is more likely to struggle handling tiles tilted at unpredictable orientations.
- **Learned**: A neural network trained on examples. Implementing this is often expensive, requiring lots of time and effort to both collect a sufficient dataset and train the policy itself. Their output is non-deterministic, but adaptable. Would handle the tilted tile pickup better if trained on similar examples.

> The pattern I took from this is to use fixed rules where the world is predictable, and learning where it is not. Finding that boundary took a few attempts.

{% include youtube.html id="Ba6jArcnLAQ" caption="Video of how the robot plays a turn in the mahjong game" %}

---

## System architecture (how it works)

{% include figure.html src="writeup/architecture.png" caption="What I built, at a glance" %}

So how does Robotto actually figure out which steps to perform and how to execute them? To play the game, some hardware is required:

- **Brain:** a PC running scripts and AI policies. This includes a vision model (YOLO) to find and name tiles, and an ACT policy that learns from recorded demonstrations how to move the arm and pick up tiles guided by footage from the wrist camera. Scripts stitch these together, using IK to hover the arm over a tile, and decide which tile to discard.
- **Eyes:** an overhead depth camera (RealSense) finds tiles; a wrist camera gives a close look for precise tile pickup. The base camera identifies tiles that are picked up and shown to it.
- **Hands:** the SO-101 follower robot arm that picks and places tiles.

{% include figure.html src="writeup/workspace.jpg" narrow=true caption="The workspace setup where I toiled my days and nights" %}

Before each game, a calibration check confirms the overhead camera maps the table to 3D positions the arm can reach. I then lay out a 13-tile hand and a two-tile-high wall to draw new tiles from each turn.

{% include figure.html src="writeup/tile-layout.jpg" caption="How the tiles are laid out for a mahjong game" %}

---

## The build & technical challenges

A good chunk of my time went on collecting data to train the vision models and ACT policies. It took far more hours of teleoperation and image annotation than I expected before they performed consistently and accurately.

It wasn't a straight line getting to this point. Here’s a high-level timeline of what got me to this point, followed by some technical challenges I encountered. If you’re interested in how the months played out in more detail, you can check out this other post [here]({{ '/build-log/' | relative_url }}).

{% include figure.html src="writeup/timeline.png" alt="How I built it: each milestone unlocked the next, June to November 2026. Late June, tile recognition (learned, YOLO): a detector trained on about 9,300 public images and about 830 photos of my own tiles; accuracy 0.90, then 0.964 by August. Late June to July, hover and calibration (geometry, IK): the arm can hover over any tile the camera sees. 3 July, first pickups (learned, ACT): demonstrations re-recorded to start 5 cm above the tile. 18 August, first win face-up (rules): the whole loop works end to end, with help on reads and stray tiles. Late August to early September, hover fixes (geometry, IK): accurate enough to place tiles, not just pick them up. Late October to November, new setup and rebuild (learned, DAgger): wheeled base, game controller, fixed layout and lighting, a base camera to read face-down tiles. 17 and 19 November, clean face-down wins: 101 of 101 pickups held on 17 November, won unattended on 19 November. Next: play with my family, once it wins 8 of 10 face-down hands clean with no wrong tile ever played unnoticed." %}

### Challenge 1: Deep-learning can be brittle

I hadn’t realized how brittle learned policies can be — they sometimes struggle to adapt and generalize to unseen environments or scenarios unlike what it was trained on.

When collecting data, I made sure to capture images across a wide range of surfaces to stress-test the vision model. I recorded tile pickups across different orientations, a variety of table positions, lighting conditions, and surfaces, while incorporating DAgger for teaching failure recovery to build a robust policy from a diverse dataset.

{% include figure.html src="writeup/green-paper-test.jpg" caption="Creating a test set for the tile recognition vision model on a surface (green paper) outside its training set" %}

I would still get surprised at how it would fail even after doing all the above. It took several learned policy iterations to surface an issue with the arm being able to pick up a tile all on its lonesome, because it’d only ever seen me pick up tiles with many other tiles around it in its dataset.

{% include figure.html src="writeup/upside-down-pickup.jpg" caption="Example of what a bamboozled robot arm can look like, inventing a new upside down tile pickup style" %}

### Challenge 2: Simulation vs. real life

I held out demonstration data from table spots the ACT policies hadn’t seen during training to check if predicted movements from a newly trained policy matched my actual demonstrations. They worked well as quick sanity checks — like spotting if the gripper closed too early above the tile — but they couldn’t predict real-world performance as physical errors compounded over time.

In fact, two policy versions tied on every offline test (scoring the policy against recorded examples on the PC, without a physical trial on the arm). Yet on the physical arm, one picked up the tile 6 out of 14 times against 11 out of 14 for the other. It happened again in November: three versions tied offline, then went 23, 25 and 27 clean out of 28 on the arm. Those arm samples are small too (the November gaps were within noise), so the lesson isn’t that the arm is a perfect judge. It’s that offline scores didn’t predict which policy would work. Now, offline tests serve only as a preliminary screen before paired physical trials on a standardized setup.

### Challenge 3: Safety guardrails

Unlike SaaS, bugs executed on a robot can physically harm you (and there were certainly some close calls during this project). Safeguards were put in place to protect against erratic, sudden movements of the robot arm. Especially when controlled by a learned model that produces non-deterministic outputs.

- **Permission first:** any command my AI coding agents run that would move the arm stops and asks me. I sit beside the arm and can stop it any time.
- **Always approach from above, always rise first:** joint speed and movement range are capped. I added this after the arm once shot straight to its zero position because its starting pose in training differed from where I ran the policy.
- **Hard limits and fail-safes:** any step that would put the robot gripper tips more than 5 mm below the table is scaled back. Impossible detections are discarded. A camera fault stops the pickup. When I’m watching, the robot asks me about tile names it’s unsure of rather than guessing.

---

## Key takeaways

### Lesson 1: Working with AI coding agents as a PM

I didn’t write the code. AI coding agents did (600+ commits, 1,300 automated tests since the late-October rebuild). My job was the PM’s job: specify what "done" means, set acceptance criteria and guardrails, and own the decisions, including which policy ships, which data counts as an exam, what "clean" means, and the rules checked before anything moves the arm.

What made it work: every hardware lesson written down as a rule the agents read before touching that code, tests that run in a simulator (software test that runs commands without the physical arm) before anything reaches the arm, and a merge queue that holds decisions for me.

{% include figure.html src="writeup/merge-process.png" caption="This process is the product of me merging seven branches in one evening with no conflicts. Bugs still got through." %}

### Lesson 2: Play found what practice couldn’t

Early play tests revealed systemic flaws that isolated practice of individual capabilities missed. It sounds straightforward, but I caught myself putting off these tests thinking it wasn’t ready for an end-to-end test yet. Instead, I would record more and more demonstrations for the ACT policy, or annotate more and more photos for the vision model.

Hundreds of practice runs made setting tiles down look solved, because each tile went back to the spot it was picked up from, and the camera-to-arm error cancelled itself out on the way there and back. But in real games, tiles travelled further, exposing a 4 to 31 mm mismatch between where the camera and the arm thought things were. Carried tiles landed 15–19 mm off target, always in the same direction. Testing the complete workflow early is crucial — component tests miss what only the full system reveals.

### Lesson 3: How I decided what “good enough” meant

When you’re in the thick of it, constantly recording new demonstrations and retraining models can feel like you’re constantly going in circles. Each fix reveals fresh edge cases that demand more training. Setting clear benchmark tests was crucial for measuring real progress and preventing old capabilities from breaking when adding new ones.

Three checks decide what happens to a new policy:

- **Iteration check (14 pairs):** more clean wins than losses against the current default, with the failure I set out to fix visibly improved.
- **Play-ready check:** at least 26 of 28 clean in one session, no spot missed twice, and no silent wrong tile name.
- **Strict paired test:** reserved for claiming “X is better”.

The one failure I treat as unacceptable is a **silent error**: a wrong tile played with nobody noticing. An assist costs a minute. A silent error means the game I thought it won, it didn’t.

---

## What’s next

At time of writing, Robotto has demonstrated it can achieve clean hands, just not consistently enough to hit the 8 clean out of 10 goal. Below I list what I'm doing to close the gap, and also hint at future scope to tackle beyond the interim goal.

- **Prove it’s repeatable.** The goal is 8 clean hands in 10, with no wrong tile ever played unnoticed. Telling close versions apart takes more trials than I expected: at 28 paired attempts, a real gain from 90% to 95% clean shows up as a clear win only 1 time in 25.
- **Improve accuracy** on the row closest to the robot, where the hand still closes about 10 mm higher on face-down tiles than face up (6 of 15 picked at one spot, against 24 of 29 face up).
- **Automate error reviews** by having a vision model flag failed grabs from wrist camera footage, which I currently inspect by hand.
- **Make it a player, not just a machine**: A screen that helps people follow the game, a voice and personality for the robot player, and a way for my grandparents to join in remotely, commanding the moves from Malaysia while the arm plays here. Success looks like my grandparents finishing a hand with Robotto without needing me to step in.
- **How I’ll test it with them.** [N=TBD] hands with my grandparents, one in person and one remote. I’ll measure how often I have to step in, how long they wait for the robot’s turn, and whether they’d choose to play another hand. The robot’s speed and its manner matter as much as accuracy to the people at the table, and I haven’t measured either yet.


---

## Closing thoughts and links

Coming from SaaS, this gave me a real appreciation for what it takes to make even a simple robotics demo reliable. I’m looking for product roles in robotics. If you build robots or play mahjong, I’d love to hear from you.

<!-- DRAFT (from review, section 2): sharper closing ask:
I’m looking for product roles in robotics, especially robots that work alongside people at home or in care. [If tailored: name the company and one sentence on why, e.g. what about their product connects to a grandparent sitting down to play a game.] If you build robots or play mahjong, I’d love to hear from you.
-->

- **Repo**: [mahjong-lekiwi](https://github.com/EthanYew/mahjong-lekiwi)
- **LinkedIn:** [https://www.linkedin.com/in/ethan-yew/](https://www.linkedin.com/in/ethan-yew/)
