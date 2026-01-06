const ADDRESS = "http://localhost:9402";

function asString(data) {
    return JSON.stringify(data);
}

function emptyCallback() {
    return;
}

function request(method, endpoint, callback = emptyCallback) {
    fetch(ADDRESS.concat(endpoint), {
        method: method,
        headers: {
            "Accept": "application/json" // change if endpoint returns text/plain or other
        }
    }).then(async response => {
        if (!response.ok) {
            const text = await response.text().catch(() => null);
            throw new Error(`HTTP ${response.status}: ${text ?? response.statusText}`);
        }
        // choose appropriate parser:
        const contentType = response.headers.get("content-type") || "";
        if (contentType.includes("application/json")) {
            return response.json();
        } else {
            return response.text();
        }
    }).then(data => {
        console.log(data);
        callback(data);
    }).catch(err => {
        console.error("Fetch error:", err);
    });
}

function requestGet(endpoint, callback = emptyCallback) {
    return request("GET", endpoint, callback);
}

function requestPut(endpoint, callback = emptyCallback) {
    return request("PUT", endpoint, callback);
}

function onButtonGet() {
    return getRequest(textInput.value, data => textOutput.value = asString(data));
}

buttonGet.addEventListener("click", onButtonGet);

// helloWorld();