# Mood and Energy Guide

Plain-language explanations of the audio features VibeTrace AI uses to score songs.
These features are numeric proxies derived from the catalog. They are useful signals,
but they are imperfect and should never be treated as objective truth about how a
song will make a specific listener feel.

## energy
Energy is a 0.0-1.0 estimate of intensity and activity. High-energy tracks feel
fast, loud, and driving; low-energy tracks feel calm and soft. Workouts usually
call for high energy (roughly 0.8+), while studying and relaxing usually call for
low energy (roughly 0.4 and below). Limitation: energy is a rough proxy and two
songs with the same energy value can still feel different.

## valence
Valence is a 0.0-1.0 estimate of musical positivity. High valence sounds happy or
cheerful; low valence sounds sad, moody, or serious. Limitation: perceived
happiness is subjective and culturally dependent, so valence is only a hint.

## tempo
Tempo is the speed of a track in beats per minute (BPM). Faster tempos (roughly
120+ BPM) suit workouts and dancing; slower tempos (below ~90 BPM) suit relaxing and
studying. Limitation: tempo alone does not determine how energetic a song feels.

## acousticness
Acousticness is a 0.0-1.0 estimate of how acoustic (versus electronic) a track is.
High acousticness suggests unplugged, organic instrumentation that many listeners
find calming and study-friendly. Limitation: acoustic does not always mean quiet.

## danceability
Danceability is a 0.0-1.0 estimate of how suitable a track is for dancing, based on
rhythm stability and beat strength. High danceability suits workouts and parties.
Limitation: danceability blends several signals and is only approximate.

## instrumentalness
Instrumentalness is a 0.0-1.0 estimate of whether a track has no vocals. High values
suggest instrumental music, which many people prefer for focus and studying because
lyrics can be distracting. Limitation: some vocal-free tracks are still busy.

## why these are imperfect proxies
Every feature above is a simplified numeric estimate, not a measurement of a
listener's experience. Personal taste, memories, context, and culture strongly shape
how music feels. VibeTrace AI uses these features because they are transparent and
explainable, but their limitations are the reason the system always shows its
reasoning, cites evidence, and reports a confidence value rather than a guarantee.
