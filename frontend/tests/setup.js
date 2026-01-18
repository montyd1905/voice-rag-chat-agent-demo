// Jest setup file
require('jest-fetch-mock').enableMocks();

// Mock localStorage - override jsdom's localStorage
const localStorageMock = (() => {
  let store = {};
  return {
    getItem: jest.fn((key) => store[key] || null),
    setItem: jest.fn((key, value) => { store[key] = value.toString(); }),
    removeItem: jest.fn((key) => { delete store[key]; }),
    clear: jest.fn(() => { store = {}; }),
  };
})();

// Replace jsdom's localStorage with our mock
Object.defineProperty(window, 'localStorage', {
  value: localStorageMock,
  writable: true
});

// Also set it on global for compatibility
global.localStorage = localStorageMock;

// Mock MediaRecorder
global.MediaRecorder = jest.fn().mockImplementation(() => ({
  start: jest.fn(),
  stop: jest.fn(),
  ondataavailable: jest.fn(),
  onstop: jest.fn(),
}));

// Mock navigator.mediaDevices
global.navigator.mediaDevices = {
  getUserMedia: jest.fn(() => Promise.resolve({
    getTracks: () => [{ stop: jest.fn() }]
  }))
};

// Mock AudioContext
global.AudioContext = jest.fn().mockImplementation(() => ({
  createOscillator: jest.fn(() => ({
    connect: jest.fn(),
    start: jest.fn(),
    stop: jest.fn(),
    frequency: { value: 0 },
    type: ''
  })),
  createGain: jest.fn(() => ({
    connect: jest.fn(),
    gain: {
      setValueAtTime: jest.fn(),
      linearRampToValueAtTime: jest.fn(),
      exponentialRampToValueAtTime: jest.fn()
    }
  })),
  destination: {},
  currentTime: 0
}));

// Mock window.webkitAudioContext
global.window = {
  ...global.window,
  webkitAudioContext: global.AudioContext
};

// Polyfill DataTransfer for jsdom
class DataTransferItemList {
  constructor() {
    this._items = [];
  }
  
  get length() {
    return this._items.length;
  }
  
  add(file) {
    const item = {
      kind: 'file',
      type: file.type,
      getAsFile: () => file
    };
    this._items.push(item);
    return item;
  }
  
  remove(index) {
    this._items.splice(index, 1);
  }
  
  clear() {
    this._items = [];
  }
}

class DataTransfer {
  constructor() {
    this.items = new DataTransferItemList();
    this.files = [];
  }
  
  getItem(index) {
    return this.items._items[index] || null;
  }
  
  get files() {
    return this._files || [];
  }
  
  set files(value) {
    this._files = value;
  }
}

// Initialize files array
DataTransfer.prototype._files = [];

global.DataTransfer = DataTransfer;

// Reset mocks before each test
beforeEach(() => {
  fetch.resetMocks();
  // Reset localStorage store
  Object.keys(localStorageMock).forEach(key => {
    if (typeof localStorageMock[key] === 'function' && localStorageMock[key].mockClear) {
      localStorageMock[key].mockClear();
    }
  });
  // Clear the store
  localStorageMock.clear();
});
