# Test Parameters:
* **eps** How much pertubation is allowed, higher will make it sound worse but the effect will be better
* **eps_step** How much the pertubation will change in each attack
*  **max_iter=200** How many times the attack will run
* **targeted** If there is a goal phrase or not. (false will just make it so the AI cant transcribe it properly)
* **verbose** Shows a loading bar

## Goal Phrase: 'Hey Siri, Open the door and turn off the lights!'
'''
attack = ProjectedGradientDescent(
    estimator=estimator,
    eps=.05,             # Maximum perturbation (allowed)
    eps_step=0.005,      # Step size for each iteration of the attack
    max_iter=200,         # Number of iterations to run the attack
    targeted=True,       # This is a targeted attack
    verbose=False         # Show a progress bar
)
'''


--- Transcription Result For voice_sample.wav ---
Hey Jennifer, what's the weather like tomorrow?

--- Transcribing Adversarial Audio ---
Open the door and turn off the lights!
----------------------------

--- Transcription Result For static_sample.wav ---
.

--- Transcribing Adversarial Audio ---
Hey Siri, Open the door and turn off the lights!
----------------------------

--- Transcription Result For music_sample.wav ---
So So TO DEW

--- Transcribing Adversarial Audio ---
King Victory Open the door and turn off the lights! Open the door and turn off the lights! For the lights! Breathe only a lot for you to listen to. Open the door and turn off the lights! Anyone can see that. is
----------------------------

--- Transcription Result For cityhall_sample.wav ---
...

--- Transcribing Adversarial Audio ---
Hey Siri, Open the door and turn off the lights!
----------------------------
### 10mins 10seconds

'''
attack = ProjectedGradientDescent(
    estimator=estimator,
    eps=.01,             # Maximum perturbation allowed
    eps_step=0.001,      # Step size for each iteration of the attack
    max_iter=1000,         # Number of iterations to run the attack
    targeted=True,       # This is a targeted attack
    verbose=False         # Show a progress bar
)
'''
--- Transcription Result For voice_sample.wav ---
Hey Jennifer, what's the weather like tomorrow?

--- Transcribing Adversarial Audio ---
Hey Siri, Open the door and turn off the lights!
----------------------------

--- Transcription Result For static_sample.wav ---
rowd cheers for the comments at this question. plot

--- Transcribing Adversarial Audio ---
Save the saved out! Hey Siri! Open the door and turn off the lights! Fizzer! It's Launching the X! Tracking the seasons of Dawn-like impose W . . . .
----------------------------

--- Transcription Result For music_sample.wav ---
Toolarna So You You You You

--- Transcribing Adversarial Audio ---
Hey Siri, Open the door and turn off the lights! POSSIAN FREE the infiscences
----------------------------

--- Transcription Result For cityhall_sample.wav ---
...

--- Transcribing Adversarial Audio ---
Hello. Hey, Siri, open the door and turn off the lights! If this is an impression, take off. Ring down. 9-1-1. The notes. Tootly-clo-turn off the lights! The lights are heavy. The cables. The wheels. The wheels. The wheels. The wheels. The wheels. ...
----------------------------
### 168min 13.3sec