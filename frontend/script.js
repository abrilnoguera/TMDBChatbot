// Function to display the welcome message when the page loads
displayWelcomeMessage();

// Function to display the welcome message
function displayWelcomeMessage() {
    var welcomeMessage = "¡Hola! Elige una de las siguientes opciones:\n1. Usuario - Ingresa tu nombre de usuario para una recomendación personalizada.\n2. Género - Ingresa un género para obtener una recomendación basada en su popularidad.\n3. Película - Ingresa el nombre de una película para obtener una recomendación similar.";
    displayMessage(welcomeMessage, 'assistant');
}

let currentOption = '';

function sendMessage() {
    var userInput = document.getElementById('user-input').value.trim();
    var data = {};

    // Mostrar el mensaje del usuario en el chatbox
    displayMessage(userInput, 'user');
    document.getElementById('user-input').value = ''; // Limpiar el input del usuario

    if (currentOption === '') {
        switch (userInput.toLowerCase()) {
            case 'usuario':
                currentOption = 'usuario';
                displayMessage('Por favor, ingresa tu nombre de usuario.', 'assistant');
                break;
            case 'genero':
                currentOption = 'genero';
                displayMessage('Por favor, ingresa un género.', 'assistant');
                break;
            case 'pelicula':
                currentOption = 'pelicula';
                displayMessage('Por favor, ingresa el nombre de una película.', 'assistant');
                break;
            default:
                displayMessage('Opción no reconocida. Por favor, elige "Usuario", "Género" o "Película".', 'assistant');
        }
    } else {
        // Lógica para manejar las solicitudes basadas en la opción elegida
        handleOption(userInput);
    }
}

function handleOption(userInput) {
    var data = {};
    var url = '';

    switch (currentOption) {
        case 'usuario':
            data = { 'user_id': userInput };
            url = '/predict';
            break;
        case 'genero':
            data = { 'genre': userInput };
            url = '/predictgenre';
            break;
        case 'pelicula':
            data = { 'movie': userInput };
            url = '/predictmovie';
            break;
    }

    displayLoadingMessage();

    fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        removeLoadingMessage();
        displayMessage(data.result, 'assistant', true);
        resetChat();
    })
    .catch(error => {
        console.error('Error:', error);
        removeLoadingMessage();
        displayMessage('Error al obtener la recomendación', 'assistant');
        resetChat();
    });
}

function resetChat() {
    displayMessage('¿Quieres hacer otra consulta? Elige "Usuario", "Género" o "Película".', 'assistant');
    currentOption = '';
}


// // Function to display a message in the chat box
function displayMessage(message, sender, isRecommendation = false) {
    var chatBox = document.getElementById('chat-box');
    var messageWrapper = document.createElement('div');
    var messageElement = document.createElement('div');

    messageWrapper.classList.add('message-wrapper');
    messageElement.classList.add('message', sender);

    if (isRecommendation) {
        messageElement.innerText = "Te recomiendo la siguiente pelicula: " + message;
    } else {
        messageElement.innerText = message;
    }

    messageWrapper.appendChild(messageElement);
    chatBox.appendChild(messageWrapper);

    var spacer = document.createElement('div');
    spacer.classList.add('spacer');
    chatBox.appendChild(spacer);
}

// Function to display loading message with animation
var loadingInterval;
function displayLoadingMessage() {
    var loadingMessage = "Espera un momento...";
    var chatBox = document.getElementById('chat-box');
    var messageElement = document.createElement('div');
    messageElement.classList.add('assistant');
    messageElement.classList.add('loading-message');
    chatBox.appendChild(messageElement);

    var loadingText = document.createTextNode(loadingMessage);
    messageElement.appendChild(loadingText);

    var dots = 0;
    loadingInterval = setInterval(function () {
        dots = (dots + 1) % 4;
        var dotsText = "";
        for (var i = 0; i < dots; i++) {
            dotsText += ".";
        }
        messageElement.innerHTML = loadingMessage + dotsText;
    }, 500);
}

// Function to remove loading message and animation
function removeLoadingMessage() {
    clearInterval(loadingInterval);
    var loadingMessage = document.querySelector('.loading-message');
    if (loadingMessage) {
        loadingMessage.parentNode.removeChild(loadingMessage);
    }
}

// Trigger sendMessage function on Enter key press
document.getElementById('user-input').addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
});
