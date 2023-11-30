// Function to display the welcome message when the page loads
displayWelcomeMessage();

// Function to display the welcome message
function displayWelcomeMessage() {
    var welcomeMessage = "¡Hola! Bienvenido al chatbot de recomendaciones de películas. Por favor, elige una de las siguientes opciones escribiendo el comando correspondiente:\n- Escribe 'Usuario' para una recomendación personalizada basada en tu nombre de usuario.\n- Escribe 'Género' para obtener una recomendación basada en un género específico.\n- Escribe 'Película' para obtener recomendaciones basadas en una película que te gusta.\n- Escribe 'Salir' para finalizar la conversación.";
    displayMessage(welcomeMessage, 'assistant');
}

let currentOption = '';

function sendMessage() {
    var userInput = document.getElementById('user-input').value.trim();

    displayMessage(userInput, 'user');
    document.getElementById('user-input').value = '';

    if (userInput.toLowerCase() === 'salir') {
        endChat();
        return;
    }

    if (currentOption === '') {
        switch (userInput.toLowerCase()) {
            case 'usuario':
                currentOption = 'usuario';
                displayMessage('Por favor, ingresa tu nombre de usuario para recibir recomendaciones personalizadas.', 'assistant');
                break;
            case 'genero':
                currentOption = 'genero';
                displayMessage('Indica un género cinematográfico para recibir recomendaciones.', 'assistant');
                break;
            case 'pelicula':
                currentOption = 'pelicula';
                displayMessage('Escribe el nombre de una película que te gusta para obtener recomendaciones similares.', 'assistant');
                break;
            default:
                displayMessage('Opción no reconocida. Por favor, escribe "Usuario", "Género", "Película" o "Salir".', 'assistant');
        }
    } else {
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


// Function to display a message in the chat box
function displayMessage(message, sender, isRecommendation = false) {
    var chatBox = document.getElementById('chat-box');
    var messageWrapper = document.createElement('div');
    var messageElement = document.createElement('div');

    messageWrapper.classList.add('message-wrapper');
    messageElement.classList.add('message', sender);

    if (isRecommendation) {
        // Comienza con el mensaje inicial
        let messageContent = "Te recomiendo las siguientes películas:<br>";
    
        // Agrega cada película en la lista como un elemento de lista
        messageContent += "<ul>";
        message.forEach((movie) => {
            messageContent += `<li>${movie}</li>`;
        });
        messageContent += "</ul>";
    
        // Establece el contenido del elemento del mensaje
        messageElement.innerHTML = messageContent;
    } else {
        messageElement.innerText = message;
    }

    messageWrapper.appendChild(messageElement);
    chatBox.appendChild(messageWrapper);

    var spacer = document.createElement('div');
    spacer.classList.add('spacer');
    chatBox.appendChild(spacer);

    scrollToBottom();
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

function endChat() {
    displayMessage("¡Gracias por usar nuestro chatbot! Hasta la próxima.", 'assistant');
    setTimeout(clearChat, 3000); // Opcional: Puedes decidir si quieres borrar el chat después de un tiempo
}

function clearChat() {
    var chatBox = document.getElementById('chat-box');
    chatBox.innerHTML = ''; // Esto borrará el contenido del chat
}

function scrollToBottom() {
    var chatBox = document.getElementById('chat-box');
    chatBox.scrollTop = chatBox.scrollHeight;
}

// Trigger sendMessage function on Enter key press
document.getElementById('user-input').addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
});
