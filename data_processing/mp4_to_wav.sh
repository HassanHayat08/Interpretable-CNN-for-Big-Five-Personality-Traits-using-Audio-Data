### Interpretable cnn for big five personaity traits using an audio data ### 
### The scripts seperates the wav data from mp4 data ###


#for file in *.mp4 format;
for file in $(ls | grep .mp4);
do 
    name1=$(ls "$file" | cut -d. -f1)
    name2=$(ls "$file" | cut -d. -f2)
    name=$name1.$name2

    # -vn: only audio out uncompressed (output.wav)
    # -ar: sampling rate 
    # -ac: no of channels (1 means mono and 2 means stereo) 
    ffmpeg -i "${file}" -ab 320k -ac 2 -ar 44100 -vn ${name}.wav;
done
