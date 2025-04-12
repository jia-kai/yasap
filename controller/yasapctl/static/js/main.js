// Update time every second
function updateTime() {
    fetch('/api/get_time')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                document.getElementById('current-time').value = data.time;
            } else {
                showError(data.error);
            }
        })
        .catch(error => showError('Failed to fetch time: ' + error));
}

// Update time immediately and then every second
updateTime();
setInterval(updateTime, 1000);

// Handle GPS update button click
document.getElementById('update-gps').addEventListener('click', () => {
    fetch('/api/update_gps', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (!data.success) {
            showError(data.error);
        }
    })
    .catch(error => showError('Failed to update GPS: ' + error));
});

// Handle mount setup button click
document.getElementById('setup-mount').addEventListener('click', () => {
    const config = {
        latitude: parseFloat(document.getElementById('latitude').value),
        longitude: parseFloat(document.getElementById('longitude').value),
        elevation: parseFloat(document.getElementById('elevation').value),
        mount_port: document.getElementById('mount-port').value,
        device_name: document.getElementById('device-name').value,
        driver: document.getElementById('driver').value
    };

    fetch('/api/setup_mount', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(config)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            window.location.href = '/control-room';
        } else {
            showError(data.error);
        }
    })
    .catch(error => showError('Failed to setup mount: ' + error));
});

// Error handling
function showError(message) {
    const modal = document.getElementById('error-modal');
    const errorMessage = document.getElementById('error-message');
    errorMessage.textContent = message;
    modal.style.display = 'block';
}

function closeErrorModal() {
    const modal = document.getElementById('error-modal');
    modal.style.display = 'none';
}

// Close modal when clicking outside
window.onclick = function(event) {
    const modal = document.getElementById('error-modal');
    if (event.target === modal) {
        modal.style.display = 'none';
    }
} 