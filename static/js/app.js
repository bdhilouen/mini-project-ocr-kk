document.addEventListener("DOMContentLoaded", function () {
    // 1. Live image preview and loading spinner on the upload page.
    var fileInput = document.getElementById("kk_image");
    var previewWrapper = document.getElementById("upload-preview-wrapper");
    var previewImage = document.getElementById("upload-preview-img");
    var uploadForm = document.getElementById("upload-form");
    var uploadSubmitBtn = document.getElementById("upload-submit-btn");

    if (fileInput && previewWrapper && previewImage) {
        fileInput.addEventListener("change", function () {
            var file = fileInput.files && fileInput.files[0];
            if (!file) {
                previewWrapper.classList.add("d-none");
                previewImage.removeAttribute("src");
                return;
            }
            var reader = new FileReader();
            reader.onload = function (evt) {
                previewImage.src = evt.target.result;
                previewWrapper.classList.remove("d-none");
            };
            reader.readAsDataURL(file);
        });
    }

    if (uploadForm && uploadSubmitBtn) {
        uploadForm.addEventListener("submit", function () {
            uploadSubmitBtn.disabled = true;
            uploadSubmitBtn.innerHTML =
                '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Memproses OCR...';
        });
    }

    // 2. Dynamic add/remove anggota rows on the confirmation page.
    var container = document.getElementById("anggota-container");
    var addBtn = document.getElementById("add-anggota-btn");
    var template = document.getElementById("anggota-row-template");
    var emptyHint = document.getElementById("anggota-empty-hint");

    function updateAnggotaNumbers() {
        if (!container) return;
        var cards = container.querySelectorAll(".anggota-card");
        cards.forEach(function (card, idx) {
            var badge = card.querySelector(".anggota-index");
            if (badge) {
                badge.textContent = "Anggota #" + (idx + 1);
            }
        });
        if (emptyHint) {
            emptyHint.classList.toggle("d-none", cards.length > 0);
        }
    }

    if (container && addBtn && template) {
        addBtn.addEventListener("click", function () {
            var fragment = template.content.cloneNode(true);
            container.appendChild(fragment);
            updateAnggotaNumbers();
            var cards = container.querySelectorAll(".anggota-card");
            var lastCard = cards[cards.length - 1];
            if (lastCard) {
                var firstInput = lastCard.querySelector("input, select");
                if (firstInput) firstInput.focus();
            }
        });

        container.addEventListener("click", function (evt) {
            var removeBtn = evt.target.closest(".remove-anggota-btn");
            if (!removeBtn) return;
            var card = removeBtn.closest(".anggota-card");
            if (card) {
                card.remove();
                updateAnggotaNumbers();
            }
        });

        updateAnggotaNumbers();
    }

    // 3. Confirmation prompt for destructive POST delete forms.
    document.addEventListener("submit", function (evt) {
        var form = evt.target;
        var confirmMsg = form.getAttribute("data-confirm");
        if (confirmMsg && !window.confirm(confirmMsg)) {
            evt.preventDefault();
        }
    });
});
