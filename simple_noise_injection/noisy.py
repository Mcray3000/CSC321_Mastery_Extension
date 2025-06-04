import soundfile as sf
import sounddevice as sd
import numpy as np
import test



def main():
    # Load Audio
    test_audio, sample_rate = sf.read("test_audio.wav")
    if test_audio.ndim > 1 or test_audio.shape[1] > 1:
        test_audio = test_audio.mean(axis=1)
    
    # Gaussian Noise
    noise_amplitude_gauss = 0.5 * np.max(np.abs(test_audio)) # Adjust noise level
    gaussian_noise = np.random.normal(0, noise_amplitude_gauss, len(test_audio))

    # Uniform Noise
    noise_amplitude_uniform = 0.5 * np.max(np.abs(test_audio)) # Adjust noise level
    uniform_noise = np.random.uniform(-noise_amplitude_uniform, noise_amplitude_uniform, len(test_audio))
    
    # Add Noise
    noisy_audio_gauss = test_audio + gaussian_noise
    noisy_audio_uniform = test_audio + uniform_noise

    # Optional: Normalize to prevent clipping if saving as int16
    noisy_audio_gauss = noisy_audio_gauss / np.max(np.abs(noisy_audio_gauss)) * 0.9
    noisy_audio_uniform = noisy_audio_uniform / np.max(np.abs(noisy_audio_uniform)) * 0.9
    
    # print("Playing original audio...")
    # sd.play(test_audio, sample_rate)
    # sd.wait()

    print("Playing Gaussian noisy audio...")
    sd.play(noisy_audio_gauss, sample_rate)
    sd.wait()

    print("Playing Uniform noisy audio...")
    sd.play(noisy_audio_uniform, sample_rate)
    sd.wait()
    
    sf.write('noisy_command_gauss.wav', noisy_audio_gauss, sample_rate)
    sf.write('noisy_command_uniform.wav', noisy_audio_uniform, sample_rate)
        
    
if __name__ == "__main__":
    main()