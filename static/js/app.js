document.addEventListener("DOMContentLoaded", function () {
    // 1. Drag-and-drop upload zone, live preview, remove button, and loading spinner.
    var fileInput = document.getElementById("kk_image");
    var dropzone = document.getElementById("upload-dropzone");
    var emptyState = document.getElementById("upload-empty-state");
    var previewWrapper = document.getElementById("upload-preview-wrapper");
    var previewImage = document.getElementById("upload-preview-img");
    var fileMeta = document.getElementById("upload-file-meta");
    var removeBtn = document.getElementById("upload-remove-btn");
    var uploadError = document.getElementById("upload-error");
    var uploadForm = document.getElementById("upload-form");
    var uploadSubmitBtn = document.getElementById("upload-submit-btn");

    var allowedExts = /\.(jpe?g|png|webp)$/i;
    var maxBytes = 10 * 1024 * 1024;

    function showUploadError(msg) {
        if (!uploadError) return;
        uploadError.textContent = msg;
        uploadError.classList.remove("d-none");
    }

    function clearUploadError() {
        if (!uploadError) return;
        uploadError.textContent = "";
        uploadError.classList.add("d-none");
    }

    function formatBytes(bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
        return (bytes / (1024 * 1024)).toFixed(2) + " MB";
    }

    function clearSelectedFile() {
        if (fileInput) fileInput.value = "";
        if (previewImage) previewImage.removeAttribute("src");
        if (fileMeta) fileMeta.textContent = "";
        if (previewWrapper) previewWrapper.classList.add("d-none");
        if (removeBtn) removeBtn.classList.add("d-none");
        if (emptyState) emptyState.classList.remove("d-none");
        if (dropzone) dropzone.classList.remove("has-file", "is-dragover");
    }

    function handleSelectedFile(file) {
        clearUploadError();
        if (!file) {
            clearSelectedFile();
            return false;
        }
        if (!allowedExts.test(file.name)) {
            clearSelectedFile();
            showUploadError("Format file tidak didukung. Gunakan JPG, JPEG, PNG, atau WEBP.");
            return false;
        }
        if (file.size > maxBytes) {
            clearSelectedFile();
            showUploadError("Ukuran file melebihi batas maksimal 10 MB.");
            return false;
        }
        var reader = new FileReader();
        reader.onload = function (evt) {
            if (previewImage) previewImage.src = evt.target.result;
            if (fileMeta) fileMeta.textContent = file.name + " (" + formatBytes(file.size) + ")";
            if (emptyState) emptyState.classList.add("d-none");
            if (previewWrapper) previewWrapper.classList.remove("d-none");
            if (removeBtn) removeBtn.classList.remove("d-none");
            if (dropzone) dropzone.classList.add("has-file");
        };
        reader.readAsDataURL(file);
        return true;
    }

    if (fileInput && dropzone) {
        dropzone.addEventListener("click", function (evt) {
            if (evt.target === fileInput || (removeBtn && removeBtn.contains(evt.target))) {
                return;
            }
            fileInput.click();
        });

        dropzone.addEventListener("keydown", function (evt) {
            if (evt.target !== dropzone) return;
            if (evt.key === "Enter" || evt.key === " ") {
                evt.preventDefault();
                fileInput.click();
            }
        });

        fileInput.addEventListener("change", function () {
            var file = fileInput.files && fileInput.files[0];
            handleSelectedFile(file);
        });

        if (removeBtn) {
            removeBtn.addEventListener("click", function (evt) {
                evt.preventDefault();
                evt.stopPropagation();
                clearSelectedFile();
                clearUploadError();
                dropzone.focus();
            });
        }

        var dragDepth = 0;
        dropzone.addEventListener("dragenter", function (evt) {
            evt.preventDefault();
            dragDepth += 1;
            dropzone.classList.add("is-dragover");
        });

        dropzone.addEventListener("dragover", function (evt) {
            evt.preventDefault();
            evt.dataTransfer.dropEffect = "copy";
        });

        dropzone.addEventListener("dragleave", function (evt) {
            evt.preventDefault();
            dragDepth = Math.max(0, dragDepth - 1);
            if (dragDepth === 0) {
                dropzone.classList.remove("is-dragover");
            }
        });

        dropzone.addEventListener("drop", function (evt) {
            evt.preventDefault();
            dragDepth = 0;
            dropzone.classList.remove("is-dragover");

            var droppedFiles = evt.dataTransfer && evt.dataTransfer.files;
            if (!droppedFiles || !droppedFiles.length) return;

            var file = droppedFiles[0];
            if (handleSelectedFile(file)) {
                var dt = new DataTransfer();
                dt.items.add(file);
                fileInput.files = dt.files;
            }
        });
    }

    if (uploadForm && uploadSubmitBtn) {
        uploadForm.addEventListener("submit", function (evt) {
            if (fileInput && (!fileInput.files || !fileInput.files.length)) {
                evt.preventDefault();
                showUploadError("Pilih atau seret file gambar Kartu Keluarga terlebih dahulu.");
                if (dropzone) dropzone.focus();
                return;
            }
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
