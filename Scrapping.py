from youtube_transcript_api import YouTubeTranscriptApi
import json

#Videos
# RFTF9tFTGAc
# r9vkHkcW3Ys
# WozDIav3TmY
# Or0akcde9ME
# BFRPHMgzsgE
# RKxna5G5LHg
# wwyC1Ixy1uY
# S0Ly5CxOC8g
# rDXPHFuodI8
# XKwv-ciNwzk
# 7WmaY46nnxw


videoId = "r9vkHkcW3Ys" #Video id, is the one afeter =v
transcript = YouTubeTranscriptApi.get_transcript(videoId, languages=['es'])#Getting the transcript from the video

with open("magistradoDataset2.json", "a", encoding="utf-8") as f:
    for i, entry in enumerate(transcript):
        json.dump({"text": entry["text"]}, f, ensure_ascii=False)
        if i < len(transcript) - 1: 
            f.write(",\n")
        else:
            f.write("\n") 