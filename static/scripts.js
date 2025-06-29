document.addEventListener("DOMContentLoaded", function() {
    const form = document.querySelector("form");
    const spinner = document.querySelector(".loading-spinner");

    form.addEventListener("submit", function() {
        spinner.style.display = "block";
    });
});
