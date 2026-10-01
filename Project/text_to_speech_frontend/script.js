const textInput = document.getElementById("textInput");
const charCount = document.getElementById("charCount");
const voiceSelect = document.getElementById("voiceSelect");
const rate = document.getElementById("rate");
const pitch = document.getElementById("pitch");
const rateValue = document.getElementById("rateValue");
const pitchValue = document.getElementById("pitchValue");

const speakBtn = document.getElementById("speakBtn");
const pauseBtn = document.getElementById("pauseBtn");
const resumeBtn = document.getElementById("resumeBtn");
const stopBtn = document.getElementById("stopBtn");
const clearBtn = document.getElementById("clearBtn");
const sampleBtn = document.getElementById("sampleBtn");

const activityIcon = document.getElementById("activityIcon");
const activityTitle = document.getElementById("activityTitle");
const activityText = document.getElementById("activityText");

const synth = window.speechSynthesis;
let voices = [];
let currentUtterance = null;

function updateCounter() {
  charCount.textContent = `${textInput.value.length} / 3000`;
}

function setActivity(icon, title, message) {
  activityIcon.textContent = icon;
  activityTitle.textContent = title;
  activityText.textContent = message;
}

function loadVoices() {
  voices = synth.getVoices();

  voiceSelect.innerHTML = "";

  if (!voices.length) {
    const option = document.createElement("option");
    option.textContent = "Default browser voice";
    option.value = "";
    voiceSelect.appendChild(option);
    return;
  }

  voices
    .slice()
    .sort((a, b) => {
      const aEnglish = a.lang.toLowerCase().startsWith("en");
      const bEnglish = b.lang.toLowerCase().startsWith("en");
      return Number(bEnglish) - Number(aEnglish);
    })
    .forEach((voice, index) => {
      const option = document.createElement("option");
      option.value = index;
      option.textContent = `${voice.name} — ${voice.lang}`;
      voiceSelect.appendChild(option);
    });
}

function speakText() {
  const text = textInput.value.trim();

  if (!text) {
    setActivity("⚠️", "No text entered", "Please type something before pressing Speak.");
    textInput.focus();
    return;
  }

  if (!("speechSynthesis" in window)) {
    setActivity("❌", "Speech not supported", "Try Chrome, Edge, or another modern browser.");
    return;
  }

  synth.cancel();

  const utterance = new SpeechSynthesisUtterance(text);
  currentUtterance = utterance;

  const selectedIndex = Number(voiceSelect.value);
  if (voices[selectedIndex]) {
    utterance.voice = voices[selectedIndex];
  }

  utterance.rate = Number(rate.value);
  utterance.pitch = Number(pitch.value);

  utterance.onstart = () => {
    setActivity("🔊", "Speaking now", "Your text is being converted into speech.");
  };

  utterance.onpause = () => {
    setActivity("⏸", "Speech paused", "Press Resume to continue.");
  };

  utterance.onresume = () => {
    setActivity("▶️", "Speech resumed", "Your browser is speaking the text.");
  };

  utterance.onend = () => {
    setActivity("✅", "Finished", "Speech conversion completed.");
  };

  utterance.onerror = (event) => {
    setActivity("❌", "Speech error", `The browser reported: ${event.error || "unknown error"}`);
  };

  synth.speak(utterance);
}

function pauseSpeech() {
  if (synth.speaking && !synth.paused) {
    synth.pause();
    return;
  }
  setActivity("ℹ️", "Nothing to pause", "Start speech first.");
}

function resumeSpeech() {
  if (synth.paused) {
    synth.resume();
    return;
  }
  setActivity("ℹ️", "Nothing to resume", "Speech is not currently paused.");
}

function stopSpeech() {
  if (synth.speaking || synth.paused) {
    synth.cancel();
    setActivity("⏹", "Stopped", "Speech playback has been stopped.");
    return;
  }
  setActivity("ℹ️", "Nothing to stop", "There is no active speech.");
}

textInput.addEventListener("input", updateCounter);

rate.addEventListener("input", () => {
  rateValue.textContent = `${Number(rate.value).toFixed(1)}×`;
});

pitch.addEventListener("input", () => {
  pitchValue.textContent = Number(pitch.value).toFixed(1);
});

sampleBtn.addEventListener("click", () => {
  textInput.value =
    "Hello! Welcome to Voxify, a simple text to speech conversion project. " +
    "This application uses the browser speech synthesis API to convert written text into spoken words.";
  updateCounter();
  setActivity("📝", "Sample loaded", "Press Speak to hear the sample text.");
});

clearBtn.addEventListener("click", () => {
  synth.cancel();
  textInput.value = "";
  updateCounter();
  setActivity("🧹", "Cleared", "Your text area is empty and ready for new text.");
  textInput.focus();
});

speakBtn.addEventListener("click", speakText);
pauseBtn.addEventListener("click", pauseSpeech);
resumeBtn.addEventListener("click", resumeSpeech);
stopBtn.addEventListener("click", stopSpeech);

if ("speechSynthesis" in window) {
  loadVoices();
  speechSynthesis.onvoiceschanged = loadVoices;
} else {
  setActivity("❌", "Speech not supported", "Use a modern browser such as Chrome or Edge.");
}

updateCounter();
