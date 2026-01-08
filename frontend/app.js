const API_BASE_URL = '/api';

let sessionId = localStorage.getItem('sessionId') || generateSessionId();
localStorage.setItem('sessionId', sessionId);

let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

// generate session ID
function generateSessionId() {
    return 'sess_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

// initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeFileUpload();
    initializeChat();
    loadDocuments();
});

// file upload
function initializeFileUpload() {
    const fileInput = document.getElementById('fileInput');
    const uploadStatus = document.getElementById('uploadStatus');
    const uploadLabel = fileInput.nextElementSibling;

    fileInput.addEventListener('change', async (e) => {
        const files = Array.from(e.target.files);
        if (files.length === 0) return;

        for (const file of files) {
            await uploadFile(file);
        }
    });

    // drag and drop
    const uploadArea = uploadLabel.closest('.upload-area');
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#764ba2';
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.borderColor = '#667eea';
    });

    uploadArea.addEventListener('drop', async (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#667eea';
        const files = Array.from(e.dataTransfer.files);
        for (const file of files) {
            await uploadFile(file);
        }
    });
}

async function uploadFile(file) {
    const uploadStatus = document.getElementById('uploadStatus');
    const documentList = document.getElementById('documentList');

    // validate file
    const validTypes = ['application/pdf', 'image/png', 'image/jpeg', 'image/tiff', 'image/bmp'];
    if (!validTypes.includes(file.type)) {
        showStatus(uploadStatus, 'error', `Invalid file type: ${file.type}`);
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        showStatus(uploadStatus, 'error', 'File too large (max 10MB)');
        return;
    }

    // show processing status
    showStatus(uploadStatus, 'processing', `Uploading ${file.name}...`);
    
    // add to document list
    const docItem = createDocumentItem(file.name, 'processing');
    documentList.appendChild(docItem);

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE_URL}/documents/upload`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Upload failed: ${response.statusText}`);
        }

        const result = await response.json();
        
        // update document item
        updateDocumentItem(docItem, result.document_id, 'completed', result.chunks, result.entities);
        showStatus(uploadStatus, 'success', `${file.name} uploaded successfully!`);
        
        // poll for status if still processing
        if (result.status === 'processing') {
            pollDocumentStatus(result.document_id, docItem);
        }
    } catch (error) {
        console.error('Upload error:', error);
        updateDocumentItem(docItem, null, 'failed');
        showStatus(uploadStatus, 'error', `Upload failed: ${error.message}`);
    }
}

function createDocumentItem(filename, status) {
    const item = document.createElement('div');
    item.className = 'document-item';
    item.innerHTML = `
        <div>
            <strong>${filename}</strong>
            <span class="status ${status}">${status}</span>
        </div>
        <div id="doc-${Date.now()}-info"></div>
    `;
    return item;
}

function updateDocumentItem(item, docId, status, chunks, entities) {
    const statusSpan = item.querySelector('.status');
    statusSpan.className = `status ${status}`;
    statusSpan.textContent = status;
    
    const infoDiv = item.querySelector('div:last-child');
    if (status === 'completed') {
        infoDiv.innerHTML = `<small>ID: ${docId} | Chunks: ${chunks} | Entities: ${entities}</small>`;
    } else if (status === 'failed') {
        infoDiv.innerHTML = `<small style="color: #dc3545;">Upload failed</small>`;
    }
}

async function pollDocumentStatus(documentId, docItem) {
    const maxAttempts = 30;
    let attempts = 0;

    const interval = setInterval(async () => {
        attempts++;
        try {
            const response = await fetch(`${API_BASE_URL}/documents/status/${documentId}`);
            if (!response.ok) return;

            const result = await response.json();
            if (result.status === 'completed' || result.status === 'failed') {
                clearInterval(interval);
                updateDocumentItem(docItem, documentId, result.status, result.chunks, result.entities);
            }
        } catch (error) {
            console.error('Status check error:', error);
        }

        if (attempts >= maxAttempts) {
            clearInterval(interval);
        }
    }, 2000);
}

async function loadDocuments() {
    try {
        const response = await fetch(`${API_BASE_URL}/documents`);
        if (!response.ok) return;

        const data = await response.json();
        const documentList = document.getElementById('documentList');
        
        data.documents.forEach(doc => {
            const item = createDocumentItem(doc.filename, doc.status);
            documentList.appendChild(item);
        });
    } catch (error) {
        console.error('Load documents error:', error);
    }
}

function showStatus(element, type, message) {
    element.className = `upload-status ${type}`;
    element.textContent = message;
    element.style.display = 'block';
    
    if (type === 'success' || type === 'error') {
        setTimeout(() => {
            element.style.display = 'none';
        }, 5000);
    }
}

// chat
function initializeChat() {
    const textInput = document.getElementById('textInput');
    const sendTextBtn = document.getElementById('sendTextBtn');
    const recordBtn = document.getElementById('recordBtn');
    const stopRecordBtn = document.getElementById('stopRecordBtn');

    sendTextBtn.addEventListener('click', sendTextMessage);
    textInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendTextMessage();
        }
    });

    recordBtn.addEventListener('click', startRecording);
    stopRecordBtn.addEventListener('click', stopRecording);
}

async function sendTextMessage() {
    const textInput = document.getElementById('textInput');
    const query = textInput.value.trim();
    
    if (!query) return;

    // add user message to chat
    addMessage('user', query);
    textInput.value = '';
    sendTextBtn.disabled = true;

    try {
        const response = await fetch(`${API_BASE_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                session_id: sessionId
            })
        });

        if (!response.ok) {
            throw new Error(`Query failed: ${response.statusText}`);
        }

        const result = await response.json();
        addMessage('assistant', result.response, result.source, result.rectified_query);
    } catch (error) {
        console.error('Query error:', error);
        addMessage('assistant', `Error: ${error.message}`, 'error');
    } finally {
        sendTextBtn.disabled = false;
    }
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            await sendVoiceMessage(audioBlob);
            stream.getTracks().forEach(track => track.stop());
        };

        mediaRecorder.start();
        isRecording = true;

        document.getElementById('recordBtn').style.display = 'none';
        document.getElementById('stopRecordBtn').style.display = 'inline-block';
        document.getElementById('recordingStatus').classList.add('active');
        document.getElementById('recordingStatus').textContent = 'Recording...';
    } catch (error) {
        console.error('Recording error:', error);
        alert('Failed to start recording. Please check microphone permissions.');
    }
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;

        document.getElementById('recordBtn').style.display = 'inline-block';
        document.getElementById('stopRecordBtn').style.display = 'none';
        document.getElementById('recordingStatus').classList.remove('active');
        document.getElementById('recordingStatus').textContent = '';
    }
}

async function sendVoiceMessage(audioBlob) {
    // show user message
    addMessage('user', '🎤 Voice message', 'voice');

    try {
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.wav');
        formData.append('session_id', sessionId);

        const response = await fetch(`${API_BASE_URL}/query/voice`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Voice query failed: ${response.statusText}`);
        }

        // check if response is audio
        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('audio')) {
            // handle audio response
            const audioBlob = await response.blob();
            const audioUrl = URL.createObjectURL(audioBlob);
            
            // get query data from headers (base64 encoded)
            const queryData = response.headers.get('X-Query-Data');
            const queryDataEncoding = response.headers.get('X-Query-Data-Encoding');
            let queryInfo = {};
            if (queryData) {
                try {
                    if (queryDataEncoding === 'base64') {
                        // Decode base64 to UTF-8 string, then parse JSON
                        const decoded = atob(queryData);
                        queryInfo = JSON.parse(decoded);
                    } else {
                        // Fallback for plain JSON (backward compatibility)
                        queryInfo = JSON.parse(queryData);
                    }
                } catch (e) {
                    console.error('Failed to parse query data:', e);
                }
            }

            addMessage('assistant', queryInfo.response || 'Audio response', queryInfo.source || 'voice', null, audioUrl);
        } else {
            // handle JSON response
            const result = await response.json();
            addMessage('assistant', result.response, result.source, result.rectified_query);
        }
    } catch (error) {
        console.error('Voice query error:', error);
        addMessage('assistant', `Error: ${error.message}`, 'error');
    }
}

function addMessage(role, content, source = null, rectifiedQuery = null, audioUrl = null) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    let messageContent = `<div>${content}</div>`;
    
    if (rectifiedQuery && rectifiedQuery !== content) {
        messageContent += `<div class="timestamp" style="font-size: 0.75em; opacity: 0.6;">Refined: ${rectifiedQuery}</div>`;
    }
    
    if (source) {
        messageContent += `<div class="source">Source: ${source}</div>`;
    }
    
    if (audioUrl) {
        messageContent += `<audio controls style="margin-top: 10px; width: 100%;"><source src="${audioUrl}" type="audio/wav"></audio>`;
    }

    messageContent += `<div class="timestamp">${new Date().toLocaleTimeString()}</div>`;

    messageDiv.innerHTML = messageContent;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

