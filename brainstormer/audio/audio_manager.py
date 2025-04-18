"""
AudioManager - Captures system audio streams (mic input and speaker output)
"""

import numpy as np
import sounddevice as sd
import threading
import queue
import time
import os
from scipy.io import wavfile
from datetime import datetime

class AudioManager:
    def __init__(self, sample_rate=16000, channels=1):
        """
        Initialize the audio manager.
        
        Args:
            sample_rate: Sampling rate for audio capture
            channels: Number of audio channels (1 for mono, 2 for stereo)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_queue = queue.Queue()
        self.is_capturing = False
        self.capture_thread = None
        
        # Get list of available devices
        self.devices = sd.query_devices()
        
        # Print all available devices to help with debugging
        self.list_audio_devices()
        
        # Use system default devices
        self.input_device = sd.default.device[0]
        self.output_device = sd.default.device[1]
        
        # Audio saving options
        self.save_audio = False
        self.audio_dir = os.path.join(os.getcwd(), "audio_recordings")
        os.makedirs(self.audio_dir, exist_ok=True)
        print(f"Manual audio chunk saving enabled. Directory: {self.audio_dir}")
    
    def print_audio_devices(self):
        """Print information about selected audio devices"""
        print(f"Selected input device [{self.input_device}]: {self.devices[self.input_device]['name']}")
        print(f"Selected output device [{self.output_device}]: {self.devices[self.output_device]['name']}")
    
    def audio_callback(self, indata, outdata, frames, callback_time, status):
        """Callback function for audio streams"""
        
        # Put captured audio in queue for transcription
        self.audio_queue.put(indata.copy())
        
    def start_capture(self):
        """Start capturing audio from system"""
        if self.is_capturing:
            print("Audio capture already running")
            return
            
        self.is_capturing = True
        
        # Start the capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        print("Audio capture started")
        
    def _capture_loop(self):
        """Main audio capture loop that runs in a separate thread"""
        try:
            # Start the input stream (microphone)
            print(f"Starting stream with input device {self.input_device} and output device {self.output_device}")
            with sd.Stream(callback=self.audio_callback, 
                          samplerate=self.sample_rate,
                          channels=self.channels,
                          device=(self.input_device, self.output_device)):
                
                print("Audio stream successfully created")
                while self.is_capturing:
                    time.sleep(0.1)  # Sleep to prevent high CPU usage
                    
        except Exception as e:
            print(f"Audio capture error: {e}")
            self.is_capturing = False
    
    def get_audio_chunk(self, duration_seconds):
        """
        Get a chunk of audio containing all data accumulated since the last call.
        
        Args:
            duration_seconds: Target duration (used for potential timeout, but not strictly enforced)
            
        Returns:
            Numpy array containing the audio data accumulated since the last call.
        """
        # Collect all available audio data from queue
        audio_data = []
        while not self.audio_queue.empty():
            try:
                # Get audio data from queue without blocking indefinitely
                chunk = self.audio_queue.get_nowait()
                audio_data.append(chunk)
            except queue.Empty:
                # Should not happen due to the while loop condition, but handle defensively
                break 
        
        if not audio_data:
            # Return an empty array with the correct shape and type if no data
            # print("No audio data in queue for this chunk.")
            return np.zeros((0, self.channels), dtype=np.float32)
            
        # Concatenate all collected chunks
        result = np.concatenate(audio_data, axis=0)
        return result
    
    def save_audio_chunk(self, audio_data, prefix="audio_"):
        """
        Save audio data to disk as a WAV file
        
        Args:
            audio_data: Numpy array containing audio data
            prefix: Prefix for the filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{prefix}{timestamp}.wav"
        filepath = os.path.join(self.audio_dir, filename)
        
        # Convert float32 audio to int16 for WAV file
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Save as WAV file
        wavfile.write(filepath, self.sample_rate, audio_int16)
        print(f"Saved audio to {filepath}")
    
    def stop_capture(self):
        """Stop audio capture"""
        self.is_capturing = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        print("Audio capture stopped")
        
    def test_microphone(self, duration=3):
        """Test if microphone is working by capturing audio for a few seconds"""
        if not self.is_capturing:
            print("Starting capture for microphone test...")
            self.start_capture()
            # Give it a moment to initialize
            time.sleep(1)
        
        print(f"Testing microphone for {duration} seconds...")
        print("Please speak into your microphone now")
        
        # Clear any existing audio in the queue
        while not self.audio_queue.empty():
            self.audio_queue.get()
            
        # Get audio for test duration
        time.sleep(duration)
        
        # Check if we received any significant audio
        chunks = []
        while not self.audio_queue.empty():
            chunks.append(self.audio_queue.get())
            
        if not chunks:
            print("No audio chunks received during test!")
            return False
            
        audio_data = np.concatenate(chunks, axis=0)
        max_level = np.abs(audio_data).max()
        mean_level = np.abs(audio_data).mean()
        
        print(f"Microphone test results:")
        print(f"  - Chunks received: {len(chunks)}")
        print(f"  - Max audio level: {max_level:.4f}")
        print(f"  - Mean audio level: {mean_level:.4f}")
        
        if max_level < 0.01:
            print("WARNING: Very low audio levels detected. Microphone may not be working correctly.")
            return False
        else:
            print("Microphone appears to be working.")
            return True
    
    def set_save_audio(self, enabled=True):
        """Enable or disable saving audio to disk"""
        self.save_audio = enabled
        status = "enabled" if enabled else "disabled"
        print(f"Audio saving {status}")

    def list_audio_devices(self):
        """List all available audio devices with their indices"""
        print("\nAvailable audio devices:")
        print("------------------------")
        for i, device in enumerate(self.devices):
            input_channels = device['max_input_channels']
            output_channels = device['max_output_channels']
            name = device['name']
            
            device_type = []
            if input_channels > 0:
                device_type.append("INPUT")
            if output_channels > 0:
                device_type.append("OUTPUT")
                
            device_info = f"[{i}] {name} ({', '.join(device_type)})"
            if i == sd.default.device[0]:
                device_info += " - DEFAULT INPUT"
            if i == sd.default.device[1]:
                device_info += " - DEFAULT OUTPUT"
                
            print(device_info)
        print("------------------------\n")
        
    def set_devices(self, input_device_id=None, output_device_id=None):
        """Set specific input and output devices by ID"""
        if input_device_id is not None:
            if input_device_id < len(self.devices) and self.devices[input_device_id]['max_input_channels'] > 0:
                self.input_device = input_device_id
                print(f"Input device set to: [{input_device_id}] {self.devices[input_device_id]['name']}")
            else:
                print(f"Invalid input device ID: {input_device_id}")
                
        if output_device_id is not None:
            if output_device_id < len(self.devices) and self.devices[output_device_id]['max_output_channels'] > 0:
                self.output_device = output_device_id
                print(f"Output device set to: [{output_device_id}] {self.devices[output_device_id]['name']}")
            else:
                print(f"Invalid output device ID: {output_device_id}")
