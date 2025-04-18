#!/usr/bin/env python3
"""
Brainstormer Agent - Main Entry Point (Async Version)
Captures audio streams and generates ideas from ongoing conversations asynchronously.
"""

import argparse
import asyncio
import numpy as np
from brainstormer.audio import AudioManager
from brainstormer.transcriber import TranscriptionService
from brainstormer.ideas import IdeaGenerator
from brainstormer.config import Config
from datetime import datetime
import os
import aiofiles # For async file operations
import logging # Use standard logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def parse_args():
    parser = argparse.ArgumentParser(description='Brainstormer - Get ideas from your conversations')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    # idea-interval is now read from config
    # parser.add_argument('--idea-interval', type=int, default=30, 
    #                     help='Interval in seconds between generating new ideas')
    return parser.parse_args()

async def summarization_loop(transcriber: TranscriptionService, config: Config, log_file_path: str):
    """Periodically summarizes the transcript log file if it gets too long."""
    summarize_interval = config.summarize_interval
    # max_tokens is no longer used here, summarization trigger might need rethinking
    # max_tokens = config.max_transcript_tokens 
    logging.info(f"Summarization loop starting. Interval: {summarize_interval}s")
    
    try:
        while True:
            await asyncio.sleep(summarize_interval)
            logging.info("Checking transcript length for summarization...")
            
            try:
                async with aiofiles.open(log_file_path, 'r', encoding='utf-8') as log_file:
                    current_transcript = await log_file.read()
                
                transcript_len_words = len(current_transcript.split())
                # Estimate token count (simple approximation, could be improved)
                transcript_len_tokens = len(current_transcript) / 4 # Rough estimate

                logging.info(f"Current transcript length: {transcript_len_words} words (~{transcript_len_tokens:.0f} tokens)")

                # --- Summarization logic needs review --- 
                # Decide if summarization is needed based on a different criterion (e.g., line count, fixed size)
                # For now, we'll comment out the token-based check.
                # if transcript_len_tokens > max_tokens:
                #     logging.info(f"Transcript potentially too long ({transcript_len_tokens:.0f} tokens), considering summarization...")
                #     # ---> Placeholder for potential future summarization logic <--- 
                #     # summary = await transcriber.summarize(current_transcript)
                #     # ... (rest of summarization file writing) ...
                # else:
                #     logging.info("Transcript length check passed (or logic disabled). No summarization triggered.")
            
            except FileNotFoundError:
                 logging.warning(f"Summarization loop: Log file not found at {log_file_path}. Skipping cycle.")
            except Exception as e:
                logging.error(f"Error during summarization cycle: {e}")
                # Avoid tight loop on error
                await asyncio.sleep(60) 

    except asyncio.CancelledError:
        logging.info("Summarization loop cancelled.")
    except Exception as e:
        logging.error(f"Error in summarization loop: {e}")
    finally:
        logging.info("Summarization loop finished.")


async def process_and_generate_loop(audio_manager: AudioManager, 
                                   transcriber: TranscriptionService, 
                                   idea_generator: IdeaGenerator, 
                                   config: Config, 
                                   log_file_path: str):
    """Handles periodic audio processing, transcription, logging, and idea generation."""
    # Get idea interval from the *main* section of the config
    idea_interval = config.config.get('main', {}).get('idea_interval', 30)
    # Get ideas log path and ensure directory exists
    ideas_log_path_str = config.llm_config.get('ideas_log_path', 'ideas_log.txt')
    ideas_log_path = os.path.join(os.getcwd(), ideas_log_path_str)
    ideas_log_dir = os.path.dirname(ideas_log_path)
    os.makedirs(ideas_log_dir, exist_ok=True)
    logging.info(f"Logging ideas to: {ideas_log_path}")

    logging.info(f"Processing and idea generation loop starting. Interval: {idea_interval}s")
    
    try:
        while True:
            await asyncio.sleep(idea_interval)
            
            # 1. Get all audio accumulated since last check
            # The duration parameter here isn't strictly used by get_audio_chunk to limit data,
            # it just drains the queue. Passing interval for consistency.
            audio_chunk = audio_manager.get_audio_chunk(duration_seconds=idea_interval)
            transcript_chunk = None
            
            # 2. Transcribe if audio data exists
            if audio_chunk is not None and audio_chunk.size > 0:
                logging.info(f"Processing audio chunk of size {audio_chunk.shape}...")
                # Transcribe asynchronously
                transcript_chunk = await transcriber.transcribe(audio_chunk)
                
                if transcript_chunk:
                    logging.info(f"Transcription: {transcript_chunk[:100]}...") # Log snippet
                    # 3. Append transcription to log file
                    try:
                        async with aiofiles.open(log_file_path, 'a', encoding='utf-8') as log_file:
                            await log_file.write(transcript_chunk + " ")
                            await log_file.flush() # Ensure it's written immediately
                    except Exception as e:
                         logging.error(f"Error writing transcript chunk to log file: {e}")
                else:
                    logging.info("Transcription resulted in empty content.")
            # else:
                # logging.debug("No new audio data for transcription cycle.") # Optional debug

            # 4. Read the *entire* current transcript from the log file
            current_transcript = ""
            try:
                async with aiofiles.open(log_file_path, 'r', encoding='utf-8') as log_file:
                    current_transcript = await log_file.read()
            except FileNotFoundError:
                 logging.warning(f"Process/Generate loop: Log file not found at {log_file_path}. Skipping idea generation.")
                 continue # Skip idea generation if file doesn't exist
            except Exception as e:
                logging.error(f"Error reading transcript log file: {e}")
                continue # Skip idea generation on read error

            # 5. Generate ideas if transcript has content
            if current_transcript and current_transcript.strip():
                transcript_len = len(current_transcript.split())
                logging.info(f"Generating ideas from transcript ({transcript_len} words)...")
                
                # Note: Summarization based on length is now handled by the summarization_loop
                # No need to check max_transcript_tokens here anymore.
                
                ideas = await idea_generator.generate(current_transcript)
                if ideas:
                    logging.info("\n--- New Ideas --- ")
                    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    try:
                        async with aiofiles.open(ideas_log_path, 'a', encoding='utf-8') as ideas_file:
                            await ideas_file.write(f"--- Ideas Generated at {timestamp_str} ---\n")
                            for idea in ideas:
                                logging.info(f"- {idea}")
                                await ideas_file.write(f"- {idea}\n")
                            await ideas_file.write("----------------------------------------\n\n")
                            await ideas_file.flush()
                    except Exception as e:
                        logging.error(f"Error writing ideas to file {ideas_log_path}: {e}")
                    logging.info("-----------------")
                else:
                    logging.info("No new ideas generated for this cycle.")
            else:
                 logging.info("Transcript log is empty or contains only whitespace. Skipping idea generation.")

    except asyncio.CancelledError:
        logging.info("Processing and idea generation loop cancelled.")
    except Exception as e:
        logging.error(f"Error in processing and idea generation loop: {e}", exc_info=True)
    finally:
        logging.info("Processing and idea generation loop finished.")


async def main_async():
    args = parse_args()
    config = Config(args.config)
    
    # --- Transcript Logging Setup --- 
    log_base_dir = config.config.get('logging', {}).get('transcript_log_dir', 'logs/transcripts')
    log_dir = os.path.join(os.getcwd(), log_base_dir)
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # Format without milliseconds
    log_file_path = os.path.join(log_dir, f"transcript_{timestamp}.txt")
    logging.info(f"Logging transcript to: {log_file_path}")
    # Create the file initially so aiofiles can open it
    try:
        with open(log_file_path, 'w', encoding='utf-8') as f: 
            f.write(f"Transcript log started at {timestamp}.\n")
    except Exception as e:
        logging.error(f"Failed to create initial log file: {e}")
        return # Exit if we can't create the log file
    # --- End Transcript Logging Setup ---
    
    # Initialize components
    audio_manager = AudioManager() # No longer needs chunk_duration from config
    transcriber = TranscriptionService(config.whisper_model)
    idea_generator = IdeaGenerator(config.llm_config)
    
    # Print settings
    logging.info(f"Using transcription model: {config.whisper_model}")
    # logging.info(f"Transcription language: {config.transcription_language}") # Removed language logging
    logging.info(f"Using idea generation model: {config.llm_config.get('model')}")
    # Get intervals from config for logging
    idea_interval = config.config.get('main', {}).get('idea_interval', 30)
    summarize_interval = config.summarize_interval
    logging.info(f"Processing/Idea generation interval: {idea_interval} seconds") 
    logging.info(f"Transcript summarization interval: {summarize_interval} seconds")
    # logging.info(f"Transcript summarization max tokens: {config.max_transcript_tokens}") # Removed max_tokens logging
    
    # Start capturing audio
    audio_manager.start_capture()
    
    process_generate_task = None
    summarization_task = None
    
    try:
        # Create and run async tasks
        process_generate_task = asyncio.create_task(
            process_and_generate_loop(audio_manager, transcriber, idea_generator, config, log_file_path)
        )
        summarization_task = asyncio.create_task(
            summarization_loop(transcriber, config, log_file_path)
        )
        
        # Wait for tasks to complete (will run until interrupted)
        await asyncio.gather(process_generate_task, summarization_task)
        
    except KeyboardInterrupt:
        logging.info("\nKeyboardInterrupt received. Stopping Brainstormer...")
    except Exception as e:
        logging.error(f"An unexpected error occurred in main_async: {e}", exc_info=True)
    finally:
        logging.info("Initiating shutdown...")
        # Cancel tasks gracefully
        if process_generate_task and not process_generate_task.done():
            logging.info("Cancelling process/generate task...")
            process_generate_task.cancel()
        if summarization_task and not summarization_task.done():
            logging.info("Cancelling summarization task...")
            summarization_task.cancel()
            
        # Wait for tasks to finish cancellation
        await asyncio.gather(process_generate_task, summarization_task, return_exceptions=True)
        
        # --- Final Processing Steps ---
        logging.info("Performing final transcription and idea generation...")
        try:
            # 1. Get remaining audio
            final_audio_chunk = audio_manager.get_audio_chunk(duration_seconds=0) # Get everything left
            final_transcript_chunk = None

            # 2. Transcribe remaining audio
            if final_audio_chunk is not None and final_audio_chunk.size > 0:
                 logging.info(f"Transcribing final audio chunk of size {final_audio_chunk.shape}...")
                 final_transcript_chunk = await transcriber.transcribe(final_audio_chunk)
                 if final_transcript_chunk:
                     logging.info(f"Final transcription chunk: {final_transcript_chunk[:100]}...")
                     # 3. Append final transcription to log file
                     try:
                         async with aiofiles.open(log_file_path, 'a', encoding='utf-8') as log_file:
                             await log_file.write(final_transcript_chunk + " ")
                             await log_file.flush()
                             logging.info(f"Appended final transcript chunk to {log_file_path}")
                     except Exception as e:
                         logging.error(f"Error writing final transcript chunk to log file: {e}")
                 else:
                     logging.info("Final transcription resulted in empty content.")
            else:
                 logging.info("No remaining audio data for final transcription.")

            # 4. Read the complete final transcript
            complete_transcript = ""
            try:
                 async with aiofiles.open(log_file_path, 'r', encoding='utf-8') as log_file:
                     complete_transcript = await log_file.read()
            except Exception as e:
                 logging.error(f"Error reading complete transcript log file for final idea generation: {e}")

            # 5. Generate final ideas if transcript exists
            if complete_transcript and complete_transcript.strip():
                 transcript_len = len(complete_transcript.split())
                 logging.info(f"Generating final ideas from complete transcript ({transcript_len} words)...")
                 final_ideas = await idea_generator.generate(complete_transcript)
                 if final_ideas:
                     logging.info("--- Final Ideas --- ")
                     timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                     ideas_log_path_str = config.llm_config.get('ideas_log_path', 'ideas_log.jsonl')
                     ideas_log_path = os.path.join(os.getcwd(), ideas_log_path_str)
                     try:
                         async with aiofiles.open(ideas_log_path, 'a', encoding='utf-8') as ideas_file:
                             await ideas_file.write(f"--- Final Ideas Generated at {timestamp_str} ---
")
                             for idea in final_ideas:
                                 logging.info(f"- {idea}")
                                 await ideas_file.write(f"- {idea}\n")
                             await ideas_file.write("----------------------------------------\n\n")
                             await ideas_file.flush()
                             logging.info(f"Wrote final ideas to {ideas_log_path}")
                     except Exception as e:
                         logging.error(f"Error writing final ideas to file {ideas_log_path}: {e}")
                     logging.info("-----------------")
                 else:
                     logging.info("No final ideas generated.")
            else:
                logging.info("Final transcript log is empty or contains only whitespace. Skipping final idea generation.")

        except Exception as e:
             logging.error(f"Error during final processing steps: {e}", exc_info=True)
        # --- End Final Processing Steps ---

        logging.info("Stopping audio capture...")
        audio_manager.stop_capture() # This is synchronous
        
        logging.info("Closing network clients...")
        # Close httpx clients used by services
        await transcriber.close()
        await idea_generator.close()
        logging.info("Brainstormer stopped.")

if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except Exception as e:
        logging.error(f"Application failed to start or run: {e}", exc_info=True) 