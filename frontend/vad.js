/**
 * Voice Activity Detection (VAD) for Aegis Studio
 * ================================================
 * Provides real-time voice activity detection for ultra-low latency voice interactions.
 * Detects when user stops speaking to trigger immediate processing.
 */

class VoiceActivityDetector {
    constructor(options = {}) {
        // Configuration
        this.silenceThreshold = options.silenceThreshold || -45; // dB
        this.silenceDuration = options.silenceDuration || 1500; // ms
        this.minSpeechDuration = options.minSpeechDuration || 500; // ms
        this.sampleRate = options.sampleRate || 16000;
        
        // State
        this.audioContext = null;
        this.analyser = null;
        this.scriptProcessor = null;
        this.mediaStream = null;
        
        this.isListening = false;
        this.isSpeaking = false;
        this.silenceStart = null;
        this.speechStart = null;
        
        // Callbacks
        this.onSpeechStart = options.onSpeechStart || (() => {});
        this.onSpeechEnd = options.onSpeechEnd || (() => {});
        this.onVolumeChange = options.onVolumeChange || (() => {});
        
        // Performance tracking
        this.stats = {
            totalSessions: 0,
            avgSpeechDuration: 0,
            avgSilenceDuration: 0,
        };
    }
    
    /**
     * Initialize and start voice activity detection
     */
    async start(stream) {
        if (this.isListening) {
            console.warn('VAD already running');
            return;
        }
        
        try {
            // Create audio context
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: this.sampleRate
            });
            
            // Create analyser
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 2048;
            this.analyser.smoothingTimeConstant = 0.3;
            
            // Connect media stream
            this.mediaStream = stream;
            const source = this.audioContext.createMediaStreamSource(stream);
            source.connect(this.analyser);
            
            // Start detection loop
            this.isListening = true;
            this.detectVoiceActivity();
            
            console.log('✓ VAD started', {
                sampleRate: this.audioContext.sampleRate,
                silenceThreshold: this.silenceThreshold,
                silenceDuration: this.silenceDuration
            });
            
            return true;
        } catch (error) {
            console.error('VAD start error:', error);
            return false;
        }
    }
    
    /**
     * Main detection loop
     */
    detectVoiceActivity() {
        if (!this.isListening) return;
        
        const bufferLength = this.analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        
        const checkAudio = () => {
            if (!this.isListening) return;
            
            // Get audio data
            this.analyser.getByteTimeDomainData(dataArray);
            
            // Calculate RMS (Root Mean Square) volume
            let sum = 0;
            for (let i = 0; i < bufferLength; i++) {
                const normalized = (dataArray[i] - 128) / 128;
                sum += normalized * normalized;
            }
            const rms = Math.sqrt(sum / bufferLength);
            
            // Convert to decibels
            const volume = rms > 0 ? 20 * Math.log10(rms) : -100;
            
            // Emit volume change
            this.onVolumeChange(volume);
            
            const now = Date.now();
            
            // Check if speaking
            if (volume > this.silenceThreshold) {
                // Voice detected
                if (!this.isSpeaking) {
                    // Speech started
                    this.isSpeaking = true;
                    this.speechStart = now;
                    this.silenceStart = null;
                    
                    console.log('Speech started');
                    this.onSpeechStart();
                }
            } else {
                // Silence detected
                if (this.isSpeaking) {
                    // Mark silence start
                    if (!this.silenceStart) {
                        this.silenceStart = now;
                    }
                    
                    // Check if silence duration exceeded
                    const silenceDuration = now - this.silenceStart;
                    const speechDuration = this.speechStart ? (now - this.speechStart) : 0;
                    
                    if (silenceDuration >= this.silenceDuration && 
                        speechDuration >= this.minSpeechDuration) {
                        // Speech ended
                        this.isSpeaking = false;
                        
                        console.log('🔇 Speech ended', {
                            speechDuration: speechDuration + 'ms',
                            silenceDuration: silenceDuration + 'ms'
                        });
                        
                        // Update stats
                        this.stats.totalSessions++;
                        this.stats.avgSpeechDuration = 
                            (this.stats.avgSpeechDuration * (this.stats.totalSessions - 1) + speechDuration) 
                            / this.stats.totalSessions;
                        this.stats.avgSilenceDuration = 
                            (this.stats.avgSilenceDuration * (this.stats.totalSessions - 1) + silenceDuration) 
                            / this.stats.totalSessions;
                        
                        this.onSpeechEnd({
                            speechDuration,
                            silenceDuration
                        });
                        
                        this.silenceStart = null;
                        this.speechStart = null;
                    }
                }
            }
            
            // Continue loop
            requestAnimationFrame(checkAudio);
        };
        
        // Start detection
        checkAudio();
    }
    
    /**
     * Stop voice activity detection
     */
    stop() {
        this.isListening = false;
        
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
        
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }
        
        console.log('VAD stopped');
    }
    
    /**
     * Adjust sensitivity
     */
    setSilenceThreshold(threshold) {
        this.silenceThreshold = threshold;
        console.log('Silence threshold updated:', threshold);
    }
    
    setSilenceDuration(duration) {
        this.silenceDuration = duration;
        console.log('Silence duration updated:', duration);
    }
    
    /**
     * Get performance statistics
     */
    getStats() {
        return {
            ...this.stats,
            isListening: this.isListening,
            isSpeaking: this.isSpeaking
        };
    }
}

/**
 * Integration with Open WebUI
 */
class AegisVoiceManager {
    constructor() {
        this.vad = null;
        this.recognition = null;
        this.isRecording = false;
        this.currentTranscript = '';
        
        this.initializeSpeechRecognition();
    }
    
    /**
     * Initialize Web Speech API
     */
    initializeSpeechRecognition() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        
        if (!SpeechRecognition) {
            console.error('Speech recognition not supported');
            return;
        }
        
        this.recognition = new SpeechRecognition();
        this.recognition.continuous = true;
        this.recognition.interimResults = true;
        this.recognition.lang = 'en-US'; // Default, can be changed
        
        this.recognition.onresult = (event) => {
            let interimTranscript = '';
            let finalTranscript = '';
            
            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript;
                
                if (event.results[i].isFinal) {
                    finalTranscript += transcript;
                } else {
                    interimTranscript += transcript;
                }
            }
            
            this.currentTranscript = finalTranscript || interimTranscript;
            
            // Display interim results
            this.updateTranscriptDisplay(this.currentTranscript, !finalTranscript);
        };
        
        this.recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            
            if (event.error === 'no-speech') {
                // Restart recognition
                setTimeout(() => {
                    if (this.isRecording) {
                        this.recognition.start();
                    }
                }, 100);
            }
        };
        
        this.recognition.onend = () => {
            if (this.isRecording) {
                // Restart if still recording
                this.recognition.start();
            }
        };
    }
    
    /**
     * Start recording with VAD
     */
    async startRecording() {
        if (this.isRecording) return;
        
        try {
            // Get microphone access
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                    sampleRate: 16000
                }
            });
            
            // Initialize VAD
            this.vad = new VoiceActivityDetector({
                onSpeechStart: () => {
                    console.log('VAD: Speech started');
                    this.showSpeakingIndicator();
                },
                onSpeechEnd: (stats) => {
                    console.log('VAD: Speech ended', stats);
                    this.hideSpeakingIndicator();
                    
                    // Auto-submit if transcript is not empty
                    if (this.currentTranscript.trim()) {
                        this.submitTranscript();
                    }
                },
                onVolumeChange: (volume) => {
                    this.updateVolumeIndicator(volume);
                }
            });
            
            await this.vad.start(stream);
            
            // Start speech recognition
            this.recognition.start();
            this.isRecording = true;
            
            console.log('Recording started with VAD');
            return true;
        } catch (error) {
            console.error('Failed to start recording:', error);
            return false;
        }
    }
    
    /**
     * Stop recording
     */
    stopRecording() {
        if (!this.isRecording) return;
        
        this.isRecording = false;
        
        if (this.vad) {
            this.vad.stop();
            this.vad = null;
        }
        
        if (this.recognition) {
            this.recognition.stop();
        }
        
        console.log('Recording stopped');
    }
    
    /**
     * Submit transcript to chat
     */
    submitTranscript() {
        if (!this.currentTranscript.trim()) return;
        
        console.log('Submitting transcript:', this.currentTranscript);
        
        // Find chat input and submit
        const chatInput = document.querySelector('textarea[placeholder*="message"]') ||
                         document.querySelector('#chat-input') ||
                         document.querySelector('textarea');
        
        if (chatInput) {
            chatInput.value = this.currentTranscript;
            
            // Trigger input event
            chatInput.dispatchEvent(new Event('input', { bubbles: true }));
            
            // Find and click submit button
            const submitButton = document.querySelector('button[type="submit"]') ||
                                document.querySelector('button[aria-label*="send"]');
            
            if (submitButton) {
                submitButton.click();
            }
            
            // Clear transcript
            this.currentTranscript = '';
            this.updateTranscriptDisplay('', false);
        }
    }
    
    /**
     * UI Updates
     */
    updateTranscriptDisplay(text, isInterim) {
        const display = document.getElementById('transcript-display');
        if (display) {
            display.textContent = text;
            display.style.opacity = isInterim ? '0.6' : '1';
        }
    }
    
    showSpeakingIndicator() {
        const indicator = document.getElementById('speaking-indicator');
        if (indicator) {
            indicator.classList.add('active');
        }
    }
    
    hideSpeakingIndicator() {
        const indicator = document.getElementById('speaking-indicator');
        if (indicator) {
            indicator.classList.remove('active');
        }
    }
    
    updateVolumeIndicator(volume) {
        const indicator = document.getElementById('volume-indicator');
        if (indicator) {
            // Convert dB to percentage (rough approximation)
            const percentage = Math.max(0, Math.min(100, (volume + 60) * 2));
            indicator.style.width = percentage + '%';
        }
    }
}

// Export for use in Open WebUI
window.AegisVoiceManager = AegisVoiceManager;
window.VoiceActivityDetector = VoiceActivityDetector;

// Auto-initialize on page load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.aegisVoice = new AegisVoiceManager();
        console.log('Aegis Voice Manager initialized');
    });
} else {
    window.aegisVoice = new AegisVoiceManager();
    console.log('Aegis Voice Manager initialized');
}