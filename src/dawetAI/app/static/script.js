// Fungsi untuk membuat bubble respons dari chatbot
function createResponseBubble() {
    const bubble = document.createElement("div");
    bubble.className = "flex items-start mb-2";
    bubble.innerHTML = `
        <div class="bg-green-500 text-white rounded-lg p-3 max-w-xs">
            <span class="response-content"></span>
        </div>
    `;
    $("#chat-area").append(bubble);
    return bubble.querySelector(".response-content");
}

// Fungsi untuk membuat bubble pesan yang dikirim oleh user
function createSendBubble(message) {
    return `
        <div class="flex items-start justify-end mb-2">
            <div class="bg-gray-200 text-gray-900 rounded-lg p-3 max-w-xs">${message}</div>
        </div>
    `;
}

// Inisialisasi WebSocket
const socket = new WebSocket(`ws://${window.location.host}/predict`);

// Event saat koneksi WebSocket terbuka
socket.onopen = function () {
    console.log("WebSocket connected");
    $("#bot-status").text("Online").removeClass("text-red-400").addClass("text-green-400");
};

// Event saat koneksi WebSocket ditutup
socket.onclose = function () {
    console.log("WebSocket disconnected");
    $("#bot-status").text("Offline").removeClass("text-green-400").addClass("text-red-400");
};

// Event saat koneksi WebSocket error
socket.onerror = function (error) {
    console.error("WebSocket error:", error);
    $("#bot-status").text("Error").removeClass("text-green-400").addClass("text-red-400");
};

// Variabel untuk menyimpan elemen respons yang sedang aktif
let currentResponseElement = null;

// Event saat menerima pesan dari server
socket.onmessage = function (e) {
    function sleep(ms) {
        var start = new Date().getTime(), expire = start + ms;
        while (new Date().getTime() < expire) { }
        return;
      }
    const data = JSON.parse(e.data);

    if (data.response) {
        // Buat bubble baru hanya jika ini adalah token pertama
        if (!currentResponseElement) {
            currentResponseElement = createResponseBubble();
        }
        // Tambahkan token ke bubble yang sedang aktif
        currentResponseElement.textContent += data.response;
        $("#chat-area").scrollTop($("#chat-area")[0].scrollHeight);
    }

    // Reset elemen respons saat streaming selesai
    if (data.end_of_response) {
        currentResponseElement = null;
    }  
};

// Fungsi untuk mengirim pesan
function sendMessage() {
    const input = $("#prompt");
    const message = input.val().trim();

    if (message) {
        // Tambahkan bubble pesan yang dikirim oleh user ke chat area
        $("#chat-area").append(createSendBubble(message));
        $("#chat-area").scrollTop($("#chat-area")[0].scrollHeight);

        // Kirim pesan melalui WebSocket
        socket.send(JSON.stringify({ message: message }));

        // Kosongkan input setelah mengirim
        input.val('');
        currentResponseElement = null; // Reset elemen respons
    }
}

// Event listener untuk tombol kirim
$("#send").on("click", sendMessage);

// Event listener untuk menekan tombol Enter di input
$("#prompt").on("keypress", function (e) {
    if (e.key === "Enter") {
        e.preventDefault(); // Mencegah form submit default
        sendMessage();
    }
});
