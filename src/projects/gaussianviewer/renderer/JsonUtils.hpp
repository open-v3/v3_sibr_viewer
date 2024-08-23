#include <curl/curl.h>
#include <sstream>
#include <picojson/picojson.hpp>

// This function will be called to write the response data
size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* userp) {
    userp->append((char*)contents, size * nmemb);
    return size * nmemb;
}

std::string fetchBuffer(const std::string& url) {
    CURL* curl;
    CURLcode res;
    std::string readBuffer;

    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();
    if(curl) {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        res = curl_easy_perform(curl);

        // Check for errors
        if(res != CURLE_OK) {
            fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
        }

        curl_easy_cleanup(curl);
    }
    curl_global_cleanup();
    return readBuffer;
}

picojson::object fetchJsonObj(const std::string& url) {
    std::string jsonBuffer = fetchBuffer(url);
    picojson::value v;
    std::string err = picojson::parse(v, jsonBuffer);
    if (!err.empty()) {
        std::cerr << "Error: " << err << std::endl;
    }
    return v.get<picojson::object>();
}