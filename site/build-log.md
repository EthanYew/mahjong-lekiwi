---
layout: default
title: "Build log: MahjongBot"
description: "How I collected data and trained the vision models and ACT policies behind MahjongBot, June to November 2026."
image: build-log/early-detection.png
permalink: /build-log/
next_url: /
next_title: "MahjongBot: teaching a robot arm to play mahjong with my family"
---

This page goes into more detail regarding how I collected data and trained vision models and ACT policies for the MahjongBot project.

**By the numbers (June — November 2026)**

- **~1,650 demonstrations** recorded (about 6 hours of arm motion across 22 datasets), including ~490 where I took over mid-pickup to correct the robot
- **~32 versions** of the pickup policy trained, 16 of them since rebuilding the setup in late October
- **2,400+ real pickups** logged in the last four weeks alone, across 290 sessions on the arm
- **~830 photos of my own tiles** to teach the vision model, plus **9,800+ photos** of tiles held up to the camera to teach the face-down reader, 2,600+ of which I checked by hand
- **967 automated tests**, 623 commits and 126 merged pull requests in four weeks, written with AI coding agents while I made the product calls
- **6 hands won** on the real arm: 1 in August, 3 face-up and 2 face-down in November

### Late June (Tile Recognition)

I started with **computer vision** to spot and identify tiles lying flat on a table using my depth camera. I **fine-tuned** a base model trained on 9k+ photos of mahjong tiles from **public datasets** on the internet, with 800+ pictures of my own tiles to build a recognition model (YOLO) that scored 0.90 mAP50 on validation images it hadn’t trained on.

It wasn’t 100% accurate, but good enough to get started. I would close the gap to about 0.964 mAP50 in August by **mining for hard examples** from every session where it slipped up, and constantly feeding these back into the dataset for further targeted training.

{% include figure.html src="build-log/early-detection.png" caption="Snapshot of early tile detection progress. You can see the Character (#C) tiles gave me a lot of trouble early on due to the more subtle differences between these tiles. This class required a lot more training examples than the others." %}

### 30 June (Hovering & Calibration)

After learning how to see tiles, Robotto needed to figure out the geometry of how to reach those tiles. I took a crash course in forward and **inverse kinematics** to get the robot hovering over any tile on the table. To translate camera pixels into table coordinates for the robot base, I stuck an **ArUco marker** on the gripper and initially posed the arm by hand, one capture at a time. (I automated this sweep later, in August.)

{% include loop.html src="calibration-sweep" narrow=true caption="Automated calibration sweep video using an ArUco marker attached to the robot gripper to map pixels in camera space (where the marker was seen) to 3D coordinates in robot space (where the gripper was relative to the robot)" %}

### 3 July (ACT Pickups)

Once the arm could hover over a desired tile, I trained an **ACT policy** to pick it up. My first attempt started demonstrations from the arm’s resting pose and failed, so I re-recorded every demonstration to start from the same spot: 5 cm directly above the tile.

{% include youtube.html id="cv2pzoez0gY" caption="Early demo of success and failure when hovering over specific tiles, and picking them up." %}

### Mid-August (First winning hand, face-up)

I let my curiosity get the better of me, and spent some time testing experimental ideas that didn’t end up panning out — like teleoperating with a **VR** headset to teach complex movements such as flipping tiles.

{% include youtube.html id="Z0RD1PVx7V0" caption="Some sidequesting done in Aug feat. my Dad, exploring teleop with Quest 3 via terra_vr to teach complex movements like flipping tiles" %}

But during this phase, I wrote the **core mahjong game rules**, and the robot successfully finished its **first winning hand** of mahjong solitaire. It wasn’t a “clean” win, as it still required some human help every now and then to nudge tiles that had strayed too far back to their starting points.

{% include loop.html src="first-win-timelapse" narrow=true caption="Timelapse of Robotto’s first messy winning mahjong hand" %}

{% include figure.html src="build-log/chicken-hand.jpg" caption="The final winning hand comprising of 2 pairs and 4 sets. This is called a **chicken hand** (雞糊 or *gai wu*) in Hong Kong/Cantonese Mahjong that is worth zero faan (points/multipliers). It’s the easiest hand to make, and not always accepted as a win in certain regional play styles. But my family accepts it, so we count this as a win." %}

### Late August — Early September (Hover Fixes)

Arm positioning geometry was still too sloppy for my liking, so I rebuilt the arm movement math from scratch by hand. Shout out to Maegan Tucker and the team behind [ECE4560](https://maegantucker.com/ECE4560/so101/). Relearning trigonometry proofs did my head in. While movements scripted in the MuJoCo simulator didn’t transfer perfectly to the real hardware, the sim-to-real gap was small enough to patch with additional calibration and correction modelling.

### Mid Oct-Nov (New setup)

After returning from a month of travel abroad, I restarted the project with a new setup and ideas for how to approach competitive play against humans.

- I mounted the arm onto a small wheeled base (LeKiwi) for extra stability. This new setup required recording new datasets and training new policies for a new setup.
- I swapped the leader arm for a game controller to teleoperate episodes, which I found a lot easier to control for my use case.
- I standardised environmental elements including tile positions, lighting and background using the same table cloth.

{% include loop.html src="gamepad-teleop" narrow=true caption="Timelapse of me teleoperating the SO-101 robot arm with a game controller. For my use case, I found this more convenient and easier to control compared to using a leader arm" %}

The LeKiwi wheeled base came with a front-facing camera, which turned out to be the key to face-down play: the arm lifts a drawn tile and shows its face to that camera, like a player sneaking a look at their cards, without ever flipping it over on the table.

{% include figure.html src="build-log/tile-layout-script.jpg" caption="Calibration script I use to consistently place tiles per a preset layout" %}

### Mid Nov (First clean winning hands, face-down)

On 17 November the robot played a whole hand to a win: 43 turns over about two hours, 101 of 101 pickups held. It drew, read and played every tile itself, and I confirmed all 56 reads without correcting any. This milestone came from targeted practice data focused specifically on its past mistakes. It drew its own face-down tiles from a “wall” of tiles stacked 2 tiles high; apart from confirming reads, my jobs were setting up the hand and wall at the start, pushing the wall forward every two turns, and checking the hand at the end.

Two days later I took the next step and let it play unattended, with no read confirmations at all. It won in 15 turns and about 47 minutes, 49 of 49 pickups held, and the hand checked out at the end. I’m still reviewing its photos of the tiles it drew and discarded before I call that win fully verified.

> Check out the [video]({{ '/#hero-video' | relative_url }}) at the top of the article showing a winning hand played to completion if you haven’t already!
