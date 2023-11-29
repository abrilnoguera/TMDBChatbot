// Function to display the welcome message when the page loads
displayWelcomeMessage();

// Function to display the welcome message
function displayWelcomeMessage() {
    var welcomeMessage = "¡Hola! Estoy aquí para recomendarte películas, si queres que te recomiende en base a un genero envia genero sino escribi tu id de usuario y te hare una recomendacion personalizada!";
    displayMessage(welcomeMessage, 'assistant');
}

let isFirstMessage = true;
let genero = false;


function sendMessage() {
    var userInput = document.getElementById('user-input').value;
    var data = {};

    if (isFirstMessage) {
        if (userInput.toLowerCase() === 'genero') {
            displayMessage(userInput, 'user');
            document.getElementById('user-input').value = '';
            displayMessage('Dime el género que te gustaría', 'assistant');
            isFirstMessage = false;
            genero = true;
            return; // Salir sin hacer una llamada a la API
        } else {
            data = { 'user_id': userInput };
        }
        isFirstMessage = false;
    } else {
        if (genero) {
            data = { 'movie': userInput };
            var url = '/predictgenre';

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
            })
            .catch(error => {
                console.error('Error:', error);
                removeLoadingMessage();
                displayMessage('Error fetching recommendation', 'assistant');
            });
            genero = false; // Reiniciar la bandera genero después de hacer la llamada a la API
        } else {
            data = { 'user_id': userInput };
        }
    }

    displayMessage(userInput, 'user'); // Mostrar el mensaje del usuario en el chatbox
    document.getElementById('user-input').value = ''; // Limpiar el input del usuario después de enviar el mensaje

    displayLoadingMessage();

    var url = '/predict';

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
    })
    .catch(error => {
        console.error('Error:', error);
        removeLoadingMessage();
        displayMessage('Error fetching recommendation', 'assistant');
    });
}


// Function to display a message in the chat box
function displayMessage(message, sender, isRecommendation = false) {
    var chatBox = document.getElementById('chat-box');
    var messageWrapper = document.createElement('div');
    var messageElement = document.createElement('div');

    messageWrapper.classList.add('message-wrapper');
    messageElement.classList.add('message', sender);

    if (isRecommendation) {
        messageElement.innerText = "Te recomiendo la siguiente peli: " + message;
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