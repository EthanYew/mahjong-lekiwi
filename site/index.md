---
layout: default
title: "MahjongBot: teaching a robot arm to play mahjong with my family"
description: "How I trained a low-cost SO-101 robot arm to play mahjong (mostly) by itself, so my family can play with it."
image: writeup/grandparents.jpg
next_url: /build-log/
next_title: "Build log: MahjongBot"
---

## The short version

I’m Ethan, a product manager based in Sydney. On a career break, I trained a low-cost, open-source robot arm (SO-101) to play mahjong (mostly) by itself. The goal is a robot my family can play mahjong with, including my grandparents when they’re back in Malaysia. This is the first of several personal projects to learn robotics by doing and pivot my career towards physical AI.

**How it works, in brief:** The arm is steered by two kinds of software. A **vision model** looks through cameras to find and name tiles. A **policy**, a neural network trained on recorded examples of me doing the task, controls the arm’s movements to pick a tile up. Scripts handle the rest. Every new version of either one is a candidate I have to test before trusting it.

**Where it stands, 19 November 2026:** **TODO** It hasn’t played with a person yet. That starts once **8 face-down hands in a row make no silent errors, and need no corrections**.

{% include youtube.html id="2sP834I__1Y" anchor="hero-video" caption="Final result: the robot reaching a winning hand from start to finish" %}

## Why mahjong

My family only has enough mahjong “kaki” (buddies) to play when my grandparents are in town from Malaysia. That happens once a year.

{% include figure.html src="writeup/grandparents.jpg" caption="Grandparents playing mahjong with my sister and I. I always get my butt kicked when they’re in town." %}

In June 2026 they were in Sydney. I’d just built two SO-101 arms: a follower that does the work and a leader I move by hand to teach it. Before they flew home, I showed them a classic pick-and-place demo of putting a ball into a container and let them remote-control it by hand (teleoperation).

I promised that one day “Robotto”, as they’d affectionately nicknamed it, would play mahjong with them. This write-up follows that promise. The first step, teaching Robotto how to play mahjong by itself, before it learns how to compete with other people.

## The mahjong game loop

> Mahjong is a game typically played with four people. Each player holds **13 tiles**. On each **turn** you **draw a tile** from the **wall** and **throw one away**, aiming to be the first to turn your 14 tiles into a winning set of groups (aka **hand**).

{% include figure.html src="writeup/game-loop.png" alt="How a hand of mahjong works. The tiles: dots, bamboo and characters numbered 1 to 9, plus winds and dragons, four copies of each. The goal: four sets and a pair, 14 tiles, where a set is a run of three in a row in one suit or a triplet of the same tile. Every turn: draw a tile so the hand has 14; if it is a winning hand, win; if not, discard one tile back to 13 and take the next turn. Robotto’s version plays alone, reads face-down tiles by lifting each drawn tile to a camera, and does not yet claim other players’ discards, score, or play against people." %}

Non-goals: no specific regional rule set, no opponents yet, simply chasing a winning hand.

### Dissecting a turn, step by step

Robotto must complete six key steps every turn in the game:

{% include figure.html src="writeup/turn-steps.png" alt="One turn, step by step: six steps, three kinds of work. 1 Find (learned): YOLO on the overhead camera spots each tile and depth turns its pixels into a 3D position; face-down tiles in hand are tracked by saved slots. 2 Get above (geometry): IK moves the hand over the tile and a correction learned from hundreds of measured hovers cleans up the error. 3 Grasp (learned): an ACT policy lowers the hand, grabs the tile and lifts it, watching through the wrist camera. 4 Read (learned): the arm shows a drawn tile to the camera on its wheeled base and a small classifier names it. 5 Decide (rules): game code picks the discard that brings the hand closest to winning. 6 Return (geometry and learned): the drawn tile fills the discard’s gap, aimed by a correction fitted from over 1,000 measured set-downs." %}

Each step in a turn is done by one of three kinds of work:

- **Geometry:** Inverse kinematics (IK) is the math that works out how each motor needs to turn to place the gripper (robot claw) at a specific spot in space — like figuring out how to move your shoulder and elbow to reach a target point with your fingers.
- **Rules:** ordinary scripted code. Deterministic, predictable output. Cheap to implement, but rigid. A scripted tile pickup motion is more likely to struggle handling tiles tilted at unpredictable orientations.
- **Learned**: A neural network trained on examples. Implementing this is often expensive, requiring lots of time and effort to both collect a sufficient dataset and train the policy itself. Their output is non-deterministic, but adaptable. Would handle the tilted tile pickup better if trained on similar examples.

> The pattern I took from this is to use fixed rules where the world is predictable, and learning where it is not. Finding that boundary took a few attempts.

{% include youtube.html id="Ba6jArcnLAQ" caption="Video of how the robot plays a turn in the mahjong game" %}

## Scoping, metrics, & quality gates

I held myself to a definition of “good enough to ship”, like I would at work. From shipping new model versions, or when we can confidently state the robot is ready for competitive play with humans.

### Defining success upfront

A solo mahjong game is roughly 23 tile picks. At 99% success per pick, about 8 games in 10 finish with no help. At 95%, only about 1 in 3 does.

> Thus the goal is: 8 clean hands in every 10 games

My headline leading metric to achieve this goal was the *clean attempt*: the tile is picked up, no tiles are bumped or knocked out of place, and no one helps (intervenes). It’s stricter than just “picked up”. Early policies picked 92% of tiles on the table, but was clean only 67% of the time, because they kept knocking neighbouring tiles. In the context of a game, this risks pushing tiles outside the arm’s reachable area, requiring me to step in and push them back into place.

### “clean” runs vs “interventions”

Though the robot is not fully autonomous, it shouldn’t require me stepping in to stop it, or rescue it either. Where the line is drawn is expanded on below:

- **Designed step**: something I am meant to do, such as setting up the wall. Not an intervention.
- **Recovery**: the robot notices its own failure and retries. Not an intervention.
- **Confirmation:** I approve a read without changing it. Not an intervention, but I count them, because each one is a person watching and waiting.
- **Assist**: I correct a misread tile label, or physically reposition a tile, so the game can continue.
- **Safety stop**: I halt the arm operation to prevent physical damage.

### How I decided what “good enough” meant

Three checks decide what happens to a new policy:

- **Iteration check (14 pairs):** more clean wins than losses against the current default, with the failure I set out to fix visibly improved.
- **Play-ready check:** at least 26 of 28 clean in one session, no spot missed twice, and no silent wrong tile name.
- **Strict paired test:** reserved for claiming “X is better”.

The one failure I treat as unacceptable is a **silent error**: a wrong tile played with nobody noticing. An assist costs a minute. A silent error means the game I thought it won, it didn’t.

### When I override my own rules

I don’t always follow the gates blindly. When I override one, I name the benefit and record the evidence and the cost. Once I broke my own rule on purpose. The tile detector that learned to see the backs of tiles face-down found 94 of 95 on an exam it had never trained on, but named face-up tiles slightly worse (96.1% vs 97.1%). I shipped it anyway, because face-down play needed backs more. I planned to win the names back with hard examples caught in play.

### Safety guardrails

Safeguards are put in place to protect against erratic, sudden movements of the robot arm. Especially when controlled by a learned model that produces non-deterministic outputs.

- **Permission first:** any command my AI coding agents run that would move the arm stops and asks me. I sit beside the arm and can stop it any time.
- **Always approach from above, always rise first:** joint speed and range are capped. I added this after the arm once shot straight to its zero position because its starting pose in training differed from where I ran the policy.
- **Hard limits and fail-safes:** any step that would put the robot gripper tips more than 5 mm below the table is scaled back. Impossible detections are discarded. A camera fault stops the pickup. When I’m watching, the robot asks me about tile names it’s unsure of rather than guessing.

## System architecture (how it works)

{% include figure.html src="writeup/architecture.png" caption="What I built, at a glance" %}

To play the game, some hardware is required:

- **Brain:** a PC running scripts and AI policies. This includes a vision model (YOLO) to find and name tiles, and an ACT policy that learns from recorded demonstrations how to move the arm and pick up tiles guided by footage from the wrist camera. Scripts stitch these together, using IK to hover the arm over a tile, and decide which tile to discard.
- **Eyes:** an overhead depth camera (RealSense) finds tiles; a wrist camera gives a close look for precise tile pickup. The base camera identifies tiles that are picked up and shown to it.
- **Hands:** the SO-101 follower robot arm that picks and places tiles.

{% include figure.html src="writeup/workspace.jpg" narrow=true caption="The workspace setup where I toiled my days and nights" %}

Before each game, a calibration check confirms the overhead camera maps the table to 3D positions the arm can reach. I then lay out a 13-tile hand and a two-tile-high wall to draw new tiles from each turn.

{% include figure.html src="writeup/tile-layout.jpg" caption="How the tiles are laid out for a mahjong game" %}

---

## The build & technical challenges

I didn’t take the straightest path to get to this point. Here’s a high-level timeline of what got us to this point, followed by some technical challenges I encountered. If you’re interested in how the months played out in more detail, you can check out this other post [here]({{ '/build-log/' | relative_url }}).

{% include figure.html src="writeup/timeline.png" alt="How I built it: each milestone unlocked the next, June to November 2026. Late June, tile recognition (learned, YOLO): a detector trained on about 9,300 public images and about 830 photos of my own tiles; accuracy 0.90, then 0.964 by August. Late June to July, hover and calibration (geometry, IK): the arm can hover over any tile the camera sees. 3 July, first pickups (learned, ACT): demonstrations re-recorded to start 5 cm above the tile. 18 August, first win face-up (rules): the whole loop works end to end, with help on reads and stray tiles. Late August to early September, hover fixes (geometry, IK): accurate enough to place tiles, not just pick them up. Late October to November, new setup and rebuild (learned, DAgger): wheeled base, game controller, fixed layout and lighting, a base camera to read face-down tiles. 17 and 19 November, clean face-down wins: 101 of 101 pickups held on 17 November, won unattended on 19 November. Next: play with my family, once it wins 8 of 10 face-down hands clean with no wrong tile ever played unnoticed." %}

### Challenge 1: Deep-learning can be brittle

I hadn’t realized how brittle learned policies can be — they sometimes struggle to adapt and generalize to unseen environments or scenarios unlike what it was trained on.

When collecting data, I made sure to capture images across a wide range of surfaces to stress-test the vision model. I recorded tile pickups across different orientations, a variety of table positions, lighting conditions, and surfaces, while incorporating DAgger for teaching failure recovery to build a robust policy from a diverse dataset.

{% include figure.html src="writeup/green-paper-test.jpg" caption="Creating a test set for the tile recognition vision model on a surface (green paper) outside its training set" %}

I would still get surprised at how it would fail even after doing all the above. It took several learned policy iterations to surface an issue with the arm being able to pick up a tile all on its lonesome, because it’d only ever seen me pick up tiles with many other tiles around it in its dataset.

{% include figure.html src="writeup/upside-down-pickup.jpg" caption="Example of what a bamboozled robot arm can look like, inventing a new upside down tile pickup style" %}

### Challenge 2: Simulation vs. real life

I held out demonstration data from table spots the ACT policies hadn’t seen during training to check if predicted movements from a newly trained policy matched my actual demonstrations. They worked well as quick sanity checks — like spotting if the gripper closed too early above the tile — but they couldn’t predict real-world performance as physical errors compounded over time.

In fact, two policy versions tied on every offline test (scoring the policy against recorded examples on the PC, without conducting a physical trial on the arm). Yet testing on the physical arm revealed one succeeded only 6 out of 14 times compared to 11 out of 14 for the other. It happened again in November: three versions tied offline, then went 23, 25 and 27 clean picks out of 28 on the arm. Now, offline tests serve only as a preliminary screen before paired physical trials on a standardized setup.

---

## Key takeaways

### Lesson 1: Working with AI coding agents as a PM

I’m not an engineer. Since rebuilding the project in late October, AI coding agents wrote the code (600+ commits, 960+ automated tests). I set direction and owned the decisions. Including which policy ships, which data counts as an exam, what “clean” means, and rules that are checked before anything moves the arm.

What made it work: every hardware lesson written down as a rule the agents read before touching that code, tests that run in a simulator (software test that runs commands without the physical arm) before anything reaches the arm, and a merge queue that holds decisions for me.

{% include figure.html src="writeup/merge-process.png" caption="This process is the product of me merging seven branches in one evening with no conflicts. Bugs still got through." %}

### Lesson 2: Play found what practice couldn’t

Early play tests revealed systemic flaws that isolated practice of individual capabilities missed. It sounds straightforward, but I caught myself putting off these tests thinking it wasn’t ready for an end-to-end test yet. Instead, I would record more and more demonstrations for the ACT policy, or annotate more and more photos for the vision model.

Hundreds of practice runs made setting tiles down look solved, because each tile went back to the spot it was picked up from, and the camera-to-arm error cancelled itself out on the way there and back. But in real games, tiles travelled further, exposing a 4 to 31 mm mismatch between where the camera and the arm thought things were. Carried tiles landed 15–19 mm off target, always in the same direction. Testing the complete workflow early is crucial — component tests miss what only the full system reveals.

---

## What’s next

- **Prove it’s repeatable.** The goal is 8 clean hands in 10, with no wrong tile ever played unnoticed. Telling close versions apart takes more trials than I expected: at 28 paired attempts, a real gain from 90% to 95% clean shows up as a clear win only 1 time in 25.
- **Improve accuracy** on the row closest to the robot, where the hand still closes slightly too high on face-down tiles (6 of 15 picked at one spot, against 24 of 29 face up).
- **Automate error reviews** by having a vision model flag failed grabs from wrist camera footage, which I currently inspect by hand.
- **Make it a player, not just a machine**: A screen that helps people follow the game, a voice and personality for the robot player, and a way for my grandparents to join in remotely, commanding the moves from Malaysia while the arm plays here. Success looks like my grandparents finishing a hand with Robotto without needing me to step in.

---

## Closing thoughts and links

Coming from SaaS, this gave me a real appreciation for what it takes to make even a simple robotics demo reliable. I’m looking for product roles in robotics. If you build robots or play mahjong, I’d love to hear from you.

- **Repo**: [mahjong-lekiwi](https://github.com/EthanYew/mahjong-lekiwi)
- **Linkedin:** [https://www.linkedin.com/in/ethan-yew/](https://www.linkedin.com/in/ethan-yew/)
