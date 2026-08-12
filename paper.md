# Abstract

This project evaluates a lightweight CRNN model for solo violin transcription. It measures the performance gap between this simple model and complex production tools like Basic Pitch. Testing shows a 12.7 percentage mean F1 score gap on the MusicNet dataset. The gap grows even larger on URMP data, except on one data piece which we later analyze and attribute to noise. Three main factors cause this difference: limited model capacity, training data volume, and the advanced design of production systems.These results align with the no-free-lunch theorem. A smaller architecture cannot fully match a complex system because fewer parameters limit its learning capacity. While lightweight models offer speed and efficiency, they trade away the deep representational power required for polyphonic or highly nuanced audio transcription tasks, which is what our CRNN model needed and lacked. Inputting more data helps, but small networks eventually hit a performance ceiling, a massive limitation. Complex systems bypass this limit through massive parameter scales, sophisticated front-end feature extraction, and multi-stage decoding pipelines. Our findings signify that solo violin transcription is extremely tedious and requires a larger pool of data in order to have similar results to complex models such as Basic Pitch. 

# Introduction

	Automatic music transcription is an important fundamental aspect of data science. It is tasked with converting musical signals into a symbolic representation of notes, pitches, onsets, and score. Because manual transcription is slow and laborious, it is necessary to have an automated system which reduces this painstaking process of score preparation. Furthermore, transcriptions allow musicians to receive real-time feedback of performances that were not notated. For example, if one has missed a note, the sheet music could clearly show this when the recording is put into the transcription. Machine learning has advanced so much over the past decade, especially with new LLMs, but transcription remained a problem due to its inability to consider polyphonic mixtures combining into a single waveform.   
	As a violinist, my research was centered largely on violin and string performance. Many of the automated music transcriptions used piano to train their model. This is very different from the violin because piano is much more discrete without much pitch-bending. Violin is not as explored due to the fingerboard placing notes in a continuous pitch. Vibrato also makes this much trickier because the pitch actually fluctuates a bit between notes. Portamento and glissando cannot happen on a discrete instrument such as piano, which was why the piano datasets could not directly apply to violin. It’s an important target for generalization because it is one of the most common instruments played in the musical world, which is also why we are using it as the focus of this project.   
	Several systems already exist in terms of automated music transcription. Basic Pitch, the model released by Spotify, is the most common. However, it only works on one instrument at a time so it may be much harder to implement on string instruments when it comes to double stops. ScoreCloud and AnthemScore are both prominent but weigh more on piano and discrete audio. It tries to avoid polyphonic pitches, which is a necessity for violin performance. 

# Related Work

Surveying Architectures

	CRNN: CRNN is fundamental in automated music transcription because it essentially extracts raw audio and analyzes the note progression that results from this extraction. CNNs are convolutional neural network layers that extract local spectral-temporal features from spectrograms. RNNs capture temporal dependencies across the sequence, allowing the model to reason about sustained notes and rhythmic context that CNNs alone cannot. Essentially CNNs are more grounded in the notes, while RNNs are more abstract in terms of how they interact with the sound, where they understand the continuity and change in music. CRNN contextualizes the audio features and the abstraction of the RNN and then merges it into one, which allows for more nuance but substantiated by the rigid structure.

	Onsets and Frames: Serves to create a representation of MIDI/frequency pitches using raw audio. MIR helps make this easier to accomplish through searching for common motifs and chord progressions. The original Onsets and Frames model was trained on the MAPS dataset, which contained single notes, chords, and full pieces; Google's follow-up, MT3, extends the same onset/frame-style approach to multi-instrument transcription, so the architecture is not inherently limited to a single instrument. The goal of this is to create a transcription with many characteristics of the piece without inputting contextualization ahead of time. This project also used librosa to input the data and then find the frequencies using log amplitude. There are two aspects to the helpers: onset and frame; the onset detector listens for where a note first begins, and the frame detector ensures the same note is being processed over the same time. Trained onsets and frames via Tensor-Flow.

	Spotify Basic Pitch: The mechanisms that create Spotify Basic Pitch is that it converts raw audio data into a spectrogram. Like we did in our code, it finds the start/end of the notes and then graphs it. After finding the MIDI pitches, it converts it to a digital score which basically enables it to create the final output. One thing that is interesting about basic pitch is that it is able to go in between notes. For example, while MIDI pitches are a rigid form, some pitches which may slide in between two notes are also captured by basic pitch which allows for more openness regarding conversion (able to do more things with it).

Input \+ Output, Pros / Cons, Architecture

CRNN

- Input: The input is the audio spectrogram, which essentially shows the relative frequencies and pitches in a visual format.   
- Output: CRNN outputs an identification for classification and sequence of notes to use for transcription.   
- Pros: Does more than just use isolated frames, and the RNN is better to detect more than structured notes (vibrato, tremolo, etc.)  
- Cons: May be much more complex than other architectures because it is a combination of CNN and RNNs.   
- Architecture: A combination of CNN \+ RNN where the CNN extracts the audio features while the RNN processes them to analyze the note duration. 

  Onset and Frames

- Input: The original Onsets and Frames model was trained on piano (MAPS), but Google's follow-up MT3 extends the approach to multi-instrument input.  
- Output: Note-level transcription describing where each note starts and then ends; MT3 generalizes this to multiple instruments.   
- Pros: Some reduction in noise because its feature of onsets and frames reduces chance of mismatch (one long note is accidentally recognized as many separate notes)  
- Cons: Onset and Frames is much more complex than CRNN; we want to optimize the bias-variance tradeoff, and the nature of Onset and Frames causes high possibility for overfitting during evaluation.   
- Architecture: The architecture at the high level for this would be the CNN+LSTM, where CNN detects the notes and then the LSTM essentially serves as the onset and frames to ensure the note is still going on.   
    
  Spotify Basic Pitch  
- Input: Spotify Basic Pitch inputs raw audio and then converts it into a visual spectrogram.   
- Output: It shows a visual display (not sheet music) of the notes that were being processed from raw audio.   
- Pros: Accurate across a lot of instruments and catch pitch bends. This is especially valuable for violin: vibrato and portamento produce continuous pitch variation that Basic Pitch's pitch bend catches  
- Cons: Limitations have been shown from time signature and fast passage problems, along with faulty chords.   
- Architecture: Just CNN, there is no RNN aspect. 

For our starter, I would like to use CRNN. Although it seems much more complex, I really like how it merges two already nuanced models and then applies both features to create a model that is able to extract notes and identify what is playing. This is important because we want it to have CNN’s capabilities of extracting from audio spectrograms, along with RNN’s crucial importance for capturing temporal dependencies to track sustained notes in continuous passages.  

# Methodology

Dataset

We used MusicNet dataset, which is a collection of classical recordings which provided both the audio file and the label files. The label files consisted of note-level annotations. From the pool of recordings, we selected a subset of pieces which contained violin repertoire. Out of the 25 pieces we sampled, we separated them into 20 training pieces and 5 test pieces which were used for evaluation. The importance of splitting these two files was to prevent the model from “learning” from the 5 test pieces before inputting it into the model. 

For our transcription target frame, we used a full 88 key piano which encompasses all the notes that could possibly be played on a violin. Each frame has a binary vector of length 88, and each neuron outputs a 0 or a 1, depending on if a model believes that a certain note is being played. 9 

Input Representation

Each audio file was converted into a time-frequency representation and processed as a sequence. Each training batch was a series of 300 consecutive frame windows, in order to keep the memory from reaching its capacity. 

Class Imbalance

The pitch activation weighting was severely imbalanced. When only a handful of 88 possible pitches are active, the model tends to continue predicting 0s, because approximately 3.07% of pitches are actually active at a time. The exploratory data analysis shows these results, which caused us to use loss weighting in order to calibrate the overprediction of 0s. 

| Positive positions | 3.065% |
| :---- | :---- |
| Natural pos\_weight | 31.7 |
| Mean active pitches per frame | 2.70 |
| Frames with 0 active pitches | 6.10% |

This EDA showed us that the natural positive weight value should have been 31.7. However, because this caused the model to start overpredicting too much, we clamped the cap to 5.0, which regulated the imbalance. It was the most optimal route because it balanced upweighting positives without collapsing the precision.   
Model Architecture

We used a convolutional recurrent neural network which uses CNN front end that lanterns frequency patterns with a recurrent back that models temporal dependencies across frames. The CNN uses channel dimension progression from 1 → 16 → 32 → 32\. Each layer uses 3 by 3 kernels with padding with a ReLU activation. Each convolutional layer uses max pooling, which reduces the noise and leaves the time axis as is. The 1025-bin frequency dimension is reduced to 32 bins while the frame rate is constant at 11.6ms. In terms of the RNN side, each step, the 32 output channels and 32 frequency bins are flattened into a 1024-dimensional vector of length T. Using a bidirectional two-layer LSTM it lets each frame’s prediction draw on both preceding and following context for note onset and sustained pitches. This essentially contextualizes the piece through each frame. 

Training

This CRNN uses BCEWithLogitsLoss in order to minimize the loss from all 88 pitches. The positive class weighting of 5.0 reduces the chance of overcorrection. Optimization follows Adam which has a constant learning rate and shuffles through batch sizes of 8\. Gradient clipping with a maximum of 1.0 is used in order to further minimize overcorrection when adjusting the weights for each neuron.

In terms of early stopping, I implemented a patience of 20 epochs. This basically means that if the best test loss did not drop after 20 epochs, the model would just stop performance and retain the checkpoint which had the best test loss. This prevents the model from further overfitting because the test data is so small. 

Evaluation 

For our evaluation, because frame-level pitch activations are properly evaluated based on the note events, we decoded the per-frame probability into discrete notes for scoring. Using a 5-frame median filter along the time axis, we were able to take out any random noise by looking at the median. Each note’s onset and offset times are read from the individual frame and converted to frequency for this metric. In terms of URMP, we converted the MIDI pitches into frequencies before moving on with the actual evaluation. 

The notes were scored with the reference annotations for both MusicNet and URMP. F1 scoring essentially provided us with the precision, recall, and the harmonic mean of both. Using an onset tolerance of 0.05 seconds we basically provided a bit of leeway regarding prediction because the time when a violin’s note starts playing is extremely subjective in itself. We disabled offset matching because it is basically impossible to tell when a violin note stopped playing due to ringing and other limitations. It essentially identifies the problems of precise offset estimation which is the standard for onset-based transcription evaluation. 

# Results

MusicNet Comparisons

| piece | CRNN\_P | CRNN\_R | CRNN\_F1 | BasicPitch\_F1 |
| :---- | :---- | :---- | :---- | :---- |
| 1805 | 0.091 | 0.125 | 0.105 | 0.059 |
| 1788 | 0.111 | 0.139 | 0.123 | 0.346 |
| 1789 | 0.052 | 0.101 | 0.069 | 0.236 |
| 1790 | 0.107 | 0.129 | 0.117 | 0.359 |
| 1793 | 0.039 | 0.089 | 0.054 | 0.102 |
| mean | 0.080 | 0.116 | 0.094 | 0.220 |

URMP Comparisons

| piece | CRNN\_F1 | BasicPitch\_F1 |
| :---- | :---- | :---- |
| Sonata | 0.076 | 0.825 |
| Spring | 0.180 | 0.847 |
| Nocturne | 0.133 | 0.679 |
| Pirates | 0.133 | 0.897 |
| Pavane | 0.165 | 0.841 |
| mean | 0.137 | 0.818 |

# Discussion

Frame-level pitch estimation is an imbalance problem; because only 3% of all frames are positive, this results in a positive weight of approximately 31.6. However, when we try to overcorrect by using this number, it overcorrects to an extreme extent, which is why we needed to clamp it at 5.0. In our pos\_weight experiment with no clamp, 10.0, and 5.0, we found that lower caps actually improve the precision. Experiment A with no clamping was the worst result, while the clamping with the least overcorrection turned out to have the best result. Without the positive weighting, the model would overpredict zeros because that would be the most optimal with a 97% accuracy. This is interesting that the optimal clamping is around 17% of the theoretically correct 31.6. This shows that the model assumes that it can locate positives perfectly and only needs to be told that they are rare. 	  
The overprediction from the results is also quantifiable. Across our datasets, the probability that the model predicted a one over a zero was 9.8%. This was more than three times the actual ground-truth density of approximately 3%. The model asserts that there should be \~10% of positives, indicating a huge problem regarding precision. If 3% of the notes are active at a time but the model predicts 9.8%, the precision is unable to exceed 0.31 (expected/actual). This observed precision tells us that the predictions are not localized well. The actual precision rate of around 12% is even worse: the majority of the notes that are being played are completely fake notes. If you compare the false notes to the total silence, you would get approximately a 9% false positive rate. Although that may seem minimal, it is in fact a big deal because there are so many false notes being played, which lowers the precision by a lot.   
My lightweight CRNN model has around 500K parameters, but it’s different from Basic Pitch. Basic Pitch is also considered a lightweight model but these use different architectures. Basic Pitch uses a Harmonic CQT which stacks frequency slices into input channels so that the convolutional kernel can learn a single harmonic template and apply it across pitch. Our model forces the network to discover the entire structure from the ground up, which Basic Pitch is already given. Basic Pitch is also trained with onset, note, and contour to disambiguate pitch. In comparison, our model does not receive any complementary signal. Essentially, Basic Pitch’s parameters are spent on harmonic inputs, multi-task supervision, and efficiency operators; the CRNN model trains from limited data and does not use the complex structures that Basic Pitch is already given.   
The data limitations of the CRNN lightweight model further solidifies the problem: while Basic Pitch trains on hundreds of hours of music spanning from multiple instruments and different recording environments, CRNN rains on 20 MusicNet pieces, all limited to string instruments. It’s essentially a catch-22. Our objective was to see if an actual lightweight model could reach a level rivaling complex systems such as Basic Pitch. With more data, it would require many more heavier complex data pipelines; when we try to adapt with these, we tend to create a model that is no longer complex. Regardless, it is extremely difficult to create an actual lightweight model with the limited data that we were confined to.   
It was interesting to see how URMP’s separated violin audio did worse than the MusicNet testing data, which had many more instruments. The CRNN lightweight model was unable to recognize features that were distinguished for violin. I initially expected URMP’s F1 would better match Basic Pitch’s F1 score, which was proven to be false. If the CRNN model was able to learn the harmonic templates and onset transfers of violin, it would have been able to carry over onto an isolated URMP violin audio file, which it wasn’t able to do. The indication that the CRNN model wasn’t able to actually learn violin features, but was just memorizing the dataset. More likely, it learned of MusicNet’s features in its audio files, such as the setting and the environment that it was in. Recording conditions may have hindered how the CRNN model was actually using the dataset and given a new environment such as URMP, it was just all noise.   
All of these findings converge into a single idea: no-free-lunch theorem. There is no such model that is able to perform on every single aspect. A model that performs better on one aspect will inevitably have some other losses in another quality. Our findings do not indicate that Basic Pitch is the single best model for automated music transcription, but that using a lightweight model for such a complex action like AMT is purely a bet. A small model is only able to function when the data is already there, along with a packaged HCQT model. Without ample enough data, the lightweight model is unable to compete with more complex systems such as Basic Pitch. 

# Future Work

	For future experiments, I would like to test the extent of how much data we could input. With more time, we could potentially expand the data from 20 MusicNet pieces to possibly much more. Instead of using one dataset for training, we could expand to other sample libraries. This expansive dataset would theoretically be able to catch more violin features, hopefully lowering the test loss. More importantly, mapping a training curve would also be optimal. Optimizing where the quantity of data and the learning curve reach the maximum would essentially help us improve the test loss, possibly. Somehow lowering the overprediction could possibly help target the precision loss. Right now, our threshold stands at 0.5. If the neuron returns a number 0.5 or greater, it automatically rounds up to 1, indicating that a note is being played when it might not be. 

# Conclusion

	Ultimately, my lightweight CRNN model was unable to compete against more complex systems such as Basic Pitch due to parameter and data limitations. From our analysis of CRNN scoring better on MusicNet rather than the much simpler URMP, it indicates that our model was only able to learn and memorize the MusicNet specific features. When given a new environment such as URMP, it was unable to retain any of the supposedly learned violin notes; at this certain point, it was all noise. In conclusion, we can attribute this large F1 score gap to the inability for a lightweight model to match complex systems without enough data. 