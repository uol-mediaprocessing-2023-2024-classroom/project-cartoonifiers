<template>
    <v-container fluid>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link rel="stylesheet" href="styles.css">
            <title>CartoonifyMe Header</title>
        </head>
        <header class="fading-header">
            <h1>CartoonifyMe!</h1>
        </header>
        <div class="selectedImageField">
            <div class="selectedImageContainer">
                <div class="loginField">
                    <div style="display: flex" v-if="isUserNameEmpty">
                        <input required placeholder="Email" v-model="loginData.email" type="email" name="email" autocomplete="email" />
                        <input required placeholder="Password" v-model="loginData.password" type="password" name="password" autocomplete="password" />
                    </div>
                    <h1 v-if="!isUserNameEmpty" style="margin-right: 15px">
                        {{ this.userName }}
                    </h1>
                    <v-btn class="clickable loginBtn" :disabled="awaitingLoginResponse" color="#d6d8e4" @click="login">
                        <v-progress-circular indeterminate color="grey lighten-5" v-if="awaitingLoginResponse"></v-progress-circular>
                        <div style="display: flex" v-else>
                            {{ this.loginButtonText }}
                        </div>
                    </v-btn>
                </div>
                <div class="selectedImageInfo">
                    <h2>Selected Image: <br /></h2>
                </div>
                <div style="display: flex">
                    <div id="loadingOverlay">Loading...</div>
                    <img class="selectedImg" v-bind:src="selectedImage.url" />
                    <div class="inputField">
                        <!-- Simple button that calls the method 'loadImages' -->
                        <button class="basicButton" @click="loadImages(cldId)">
                            Load Images
                        </button>
                        <button class="basicButton" @click="getCartoon(selectedImage.id)">
                            Cartoonify!
                        </button>
                        <button class="basicButton" @click="cartoonAI(selectedImage.id)">
                            Cartoonif-AI!
                        </button>
                        <div>
                            <h3>{{ imageInfo.name }}<br /></h3>
                        </div>
                        <div>
                                <v-slider label="Contrast" v-model="sliderGrid" min="3" max="21" step="2" thumb-label></v-slider>
                            </div>
                            <div>
                                <v-slider label="Edges" v-model="sliderEdge" min="5" max="33" step="2" thumb-label></v-slider>
                            </div>
                            <div>
                                <v-slider label="Colors" v-model="sliderK" min="8" max="128" step="8" thumb-label></v-slider>
                            </div>
                            <div>
                                <v-slider label="Sharpness" v-model="sliderBila" min="5" max="21" step="2" thumb-label></v-slider>
                            </div>
                            <div>
                                <v-slider label="Iterations" v-model="sliderIter" min="1" max="10" step="1" thumb-label></v-slider>
                            </div>
                    </div>
                </div>
            </div>
        </div>
        <div class="imageGalleryField">
            <div>
                <v-row>
                    <v-col v-for="n in galleryImageNum" :key="n" class="d-flex child-flex" cols="2">
                        <v-img :src="currentGallery[n - 1].url" aspect-ratio="1" max-height="200" max-width="200" class="grey lighten-2" @click="updateSelected(currentGallery[n - 1].id)">
                            <template v-slot:placeholder>
                                <v-row class="fill-height ma-0" align="center" justify="center">
                                    <v-progress-circular indeterminate color="grey lighten-5"></v-progress-circular>
                                </v-row>
                            </template>
                        </v-img>
                    </v-col>
                </v-row>
            </div>
            <button class="loadMoreBtn" @click="$emit('loadMore')">Load more</button>
        </div>
    </v-container>
</template>

<script>
export default {
    name: "HomePage",

    data() {
        return {
            // User related data
            cldId: "",
            userName: "",
            isLoggedIn: false,
            loginData: {
                email: "",
                password: ""
            },
            awaitingLoginResponse: false,

            // Image related data
            imageInfo: {
                name: "",
                avgColor: ""
            },

            // UI related
            loginButtonText: "LOGIN",
            sliderGrid:9,
            sliderEdge:15,
            sliderK:16,
            sliderIter:5,
            sliderBila:9,
        };
    },

    props: {
        selectedImage: Object,
        currentGallery: Array,
    },

    methods: {
        // --- IMAGE RELATED METHODS ---

        // Emit a loadImages event.
        loadImages() {
            this.$emit("loadImages", this.cldId);
        },

        // Emit a updateSelected event with the ID of the selected image.
        // This method is called when the user clicks/selects an image in the gallery of loaded images.
        updateSelected(selectedId) {
            this.$emit("updateSelected", selectedId, this.cldId);
        },

        // Emit a getCartoon event with the ID of the selected image.
        getCartoon(selectedId) {
            var sliderValues = "["+this.sliderGrid+","+this.sliderEdge+","+this.sliderK+","+this.sliderIter+","+this.sliderBila+"]";
            this.$emit("getCartoon", selectedId, this.cldId, sliderValues);
        },

        // Emit a cartoonAI event with the ID of the selected image.
        cartoonAI(selectedId) {
            this.$emit("cartoonAI", selectedId, this.cldId);
        },

        // --- AUTHENTICATION RELATED METHODS ---

        // Send a login request to the CEWE API test server.
        // If the user is already logged in, send a logout request instead.
        async login() {
            if (this.isLoggedIn) {
                this.logout();
                return;
            }

            if (this.awaitingLoginResponse) return;
            this.awaitingLoginResponse = true;

            const requestOptions = this.getLoginRequestOptions();
            const response = await this.sendLoginRequest(requestOptions);

            if (response) {
                this.handleLoginResponse(response);
            }

            this.awaitingLoginResponse = false;
        },

        // Helper method called by login(), logs out the user.
        // Also resets saved website data.
        async logout() {
            if (!this.isLoggedIn) return;

            const response = await this.sendLogoutRequest();
            this.handleLogoutResponse(response);
        },

        // Helper method for saving user data in the browsers local storage.
        handleLoginResponse(response) {
            this.cldId = response.session.cldId;
            this.userName = response.user.firstname;
            this.isLoggedIn = true;

            localStorage.cldId = this.cldId;
            localStorage.userName = this.userName;
            localStorage.isLoggedIn = this.isLoggedIn;
        },

        // Helper method for clearing user data from the browsers local storage.
        handleLogoutResponse() {
            localStorage.cldId = "";
            localStorage.userName = "";
            localStorage.isLoggedIn = false;
            this.resetData();
        },

        // Helper method for resetting saved data.
        resetData() {
            this.cldId = "";
            this.isLoggedIn = false;
            this.userName = "";
            this.loginData = {
                email: "",
                password: ""
            };
            this.imageInfo = {
                name: "",
            };
            this.awaitingLoginResponse = false;
            this.$emit("resetGallery");
        },

        // --- REQUEST HANDLERS ---

        getLoginRequestOptions() {
            return {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    clientVersion: "0.0.1-medienVerDemo",
                    apiAccessKey: "6003d11a080ae5edf4b4f45481b89ce7",
                },
                body: JSON.stringify({
                    login: this.loginData.email,
                    password: this.loginData.password,
                    deviceName: "Medienverarbeitung CEWE API Demo",
                }),
            };
        },

        async sendLoginRequest(requestOptions) {
            let status = 0;
            try {
                const response = await fetch("https://cmp.photoprintit.com/api/account/session/", requestOptions);
                status = response.status;
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            } catch (error) {
                this.handleRequestError(error, status);
                return null;
            }
        },

        async sendLogoutRequest() {
            const requestOptions = {
                method: "DELETE",
                headers: {
                    cldId: this.cldId,
                    clientVersion: "0.0.1-medienVerDemo",
                },
            };

            try {
                const response = await fetch("https://cmp.photoprintit.com/api/account/session/?invalidateRefreshToken=true", requestOptions);
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response;
            } catch (error) {
                this.handleRequestError(error);
                return null;
            }
        },

        handleRequestError(error, status = 0) {
            console.error("Request failed:", error);
            if (status === 500 || status === 405) {
                this.displayError("Internal error occurred, please try again later.");
            } else if (status >= 400 && status < 500) {
                this.displayError("Entered credentials are incorrect or the request was not properly formatted.");
            } else {
                this.displayError("Something went wrong, please try again.");
            }
        },

        displayError(message) {
            alert(message);
        },

    },

    computed: {
        /*
            The numer of images within currentGallery can dynamically change after the DOM is loaded. Since the size of the image gallery depends on it,
            it's important for it to be updated within the DOM aswell. By using computed values this is not a problem since Vue updates the DOM in accordance wit them.
        */
        galleryImageNum() {
            return this.currentGallery.length;
        },

        isUserNameEmpty: function () {
            return this.userName == "";
        },
    },

    watch: {

        // Watcher function for updating the displayed image information.
        selectedImage: function () {
            this.imageInfo = {
                name: "Name: " + this.selectedImage.name,
                avgColor: "Average color: " + this.selectedImage.avgColor,
            };
        },

        // Watcher function for updating login button text.
        isLoggedIn(isLoggedIn) {
            if (isLoggedIn) {
                this.loginButtonText = "LOGOUT";
            } else {
                this.loginButtonText = "LOGIN";
            }
        },
    },

    mounted() {
        // Load from local storage
        if (localStorage.isLoggedIn === "true") {
            this.cldId = localStorage.cldId;
            this.userName = localStorage.userName;
            this.isLoggedIn = true;
        }
    },
};
</script>

<style scoped>

body {
    margin: 0;
    padding: 0;
    font-family: 'Arial', sans-serif;
}

.fading-header {
    background: linear-gradient(to right, #085b92, #ffffff);
    color: #fff;
    text-align: left;
    padding: 20px;
}

h1 {
    margin: 0;
    font-size: 2em;
    letter-spacing: 2px;
}

.selectedImageField {
    position: relative;
    display: flex;
    flex-direction: row;
    background-color: rgb(249, 251, 255);
    border-radius: 10px;
    box-shadow: 0 10px 10px 10px rgba(0, 0, 0, 0.1);
    color: black;
    padding: 1%;
}

.imageGalleryField {
    display: flex;
    flex-direction: column;
    background-color: rgb(249, 251, 255);
    border-radius: 10px;
    box-shadow: 0 10px 10px 10px rgba(0, 0, 0, 0.1);
    color: black;
    padding: 1%;
    margin-top: 1%;
    max-height: 600px;
    overflow-y: auto;
}

.selectedImg {
    max-width: 500px;
    max-height: 500px;
}

.selectedImageInfo {
    margin-left: 10px;
}

.basicButton {
    background-color: rgb(226, 215, 215);
    padding: 0px 4px 0px 4px;
    margin-right: 5px;
    border-radius: 3px;
    width: 150px;
}

.idInput {
    margin-right: 8px;
    border: 1px solid #000;
    border-radius: 3px;
}

.loginField {
    display: flex;
    margin-left: auto;
    margin-bottom: 12px;
}

.loginField * {
    margin: 0px 5px 0px 5px;
}

.loginField * input {
    border: 1px solid #000;
    border-radius: 3px;
}

.inputField {
    display: flex;
    flex-direction: column;
    margin-left: 10px;
    width: 400px;
}

.inputField * {
    margin: 5px 0px 5px 0px;
}

.loadMoreBtn {
    background-color: #a7a7a7;
    border-radius: 6px;
    padding-left: 5px;
    padding-right: 5px;
    width: 100px;
    align-self: center;
    margin-top: 10px;
}

.loader {
    border: 8px solid #f3f3f3;
    border-top: 8px solid #3498db;
    border-radius: 50%;
    width: 50px;
    height: 50px;
    animation: spin 1s linear infinite;
    position: absolute;
    top: 50%;
    left: 50%;
    margin-top: -25px;
    margin-left: -25px;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

#loadingOverlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(255, 255, 255, 0.8); /* Semi-transparent white background */
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000; /* Ensure it appears above other elements */
  display: none; /* Initially hide the loading overlay */
}

</style>
