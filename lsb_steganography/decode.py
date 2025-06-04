import wave
import numpy as np 

DELIMITER = '1111111111111110'

def binary_to_text(binary_string):
    """Converts a binary string back to text."""
    text = ""
    for i in range(0, len(binary_string), 8):
        byte = binary_string[i:i+8]
        if len(byte) < 8:
            # This might happen if the delimiter was not found or data is corrupted
            # Or if the binary_string length is not a multiple of 8 after delimiter removal
            # print(f"Warning: Incomplete byte '{byte}' at end of binary string.")
            break
        try:
            text += chr(int(byte, 2))
        except ValueError:
            # print(f"Warning: Could not convert byte '{byte}' to character.")
            continue # Or handle error as appropriate
    return text

def decode_lsb(stego_wav_path):
    try:
        with wave.open(stego_wav_path, mode='rb') as wf:
            n_frames = wf.getnframes()
            frames_bytes = bytearray(wf.readframes(n_frames)) # [1] for bytearray conversion

            extracted_bits_list = [] # Use a list to append bits
            
            # Iterate through each byte of the audio frame data
            for i in range(len(frames_bytes)):
                # Extract the LSB from the current byte and append as a string '0' or '1'
                extracted_bits_list.append(str(frames_bytes[i] & 1)) # [1]

                # Check for the delimiter efficiently
                # Only check if we have enough bits to form the delimiter
                if len(extracted_bits_list) >= len(DELIMITER):
                    # Form a string only from the last part of the list that could be the delimiter
                    potential_delimiter_str = "".join(extracted_bits_list)
                    
                    if potential_delimiter_str == DELIMITER:
                        # Delimiter found. Remove it from the list of extracted bits.
                        extracted_bits_list = extracted_bits_list
                        break # Exit the loop as the message end is found
            

            binary_message = "".join(extracted_bits_list)
            if not binary_message:
                print("No message bits extracted or delimiter found immediately.")
                return ""
                
            secret_message = binary_to_text(binary_message)
            print(f"Decoded message: {secret_message}")
            return secret_message

    except FileNotFoundError:
        print(f"Error: Stego file '{stego_wav_path}' not found.")
        return None
    except wave.Error as e:
        print(f"Wave error opening or reading '{stego_wav_path}': {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during decoding: {e}")
        return None

def main():
    decoded_text = decode_lsb("output.wav") 
    if decoded_text is not None:
        print(f"Main function received: '{decoded_text}'")
    else:
        print("Main function: Decoding failed.")
        
if __name__ == "__main__":
    main()