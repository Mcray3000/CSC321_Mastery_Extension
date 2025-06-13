import wave
import soundfile as sf
import sounddevice as sd
import numpy as np

DELIMITER = '1111111111111111'


def text_to_binary(text):
    """Converts a string to its binary representation."""
    binary_message = ''.join(format(ord(char), '08b') for char in text)
    return binary_message



def main():
    # Load Audio
    test_audio, sample_rate = sf.read("my_original_command.wav")
    if test_audio.ndim > 1:
        test_audio = test_audio.mean(axis=1)
    # Example:
    secret_command = "Hey Siri call McCay Ruddick"
    binary_secret_command = text_to_binary(secret_command)
    try:
        # Prepare the secret message
        binary_message = binary_secret_command + DELIMITER
        message_bits = list(binary_message) # List of '0' or '1'
        message_len = len(message_bits)
        bit_index = 0

        # Open the carrier WAV file
        with wave.open("my_original_command.wav", mode='rb') as wf:
            params = wf.getparams()
            n_channels, sampwidth, framerate, n_frames, comptype, compname = params

            frames_bytes = bytearray(wf.readframes(n_frames)) # Read all frames as a mutable byte array [3]

            if message_len > len(frames_bytes):
                raise ValueError("Message is too long to fit in the carrier audio.")

            # Modify LSBs
            for i in range(message_len): # Iterate for each bit of the message
                if bit_index < message_len:
                    # Get the current byte from audio frames
                    audio_byte = frames_bytes[i]

                    # Get the current bit from the secret message
                    message_bit = int(message_bits[bit_index])

                    # Clear the LSB of the audio byte
                    modified_byte = audio_byte & 0xFE # AND with 11111110

                    # Set the LSB of the audio byte to the message bit
                    if message_bit == 1:
                        modified_byte = modified_byte | 0x01 # OR with 00000001

                    frames_bytes[i] = modified_byte
                    bit_index += 1
                else:
                    break # All message bits embedded

        # Write the modified frames to a new WAV file
        with wave.open("output.wav", mode='wb') as out_wf:
            out_wf.setparams(params)
            out_wf.writeframes(frames_bytes)
        print(f"Message encoded into output.wav")

    except FileNotFoundError:
        print(f"Error: Carrier file not found.")
    except Exception as e:
        print(f"An error occurred during encoding: {e}")
            
    
if __name__ == "__main__":
    main()