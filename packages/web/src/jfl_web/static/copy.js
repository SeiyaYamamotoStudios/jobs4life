/* "Copy to clipboard" for a written CV or cover letter.
 *
 * Enhancement only. Without this file the text still sits in a readonly
 * textarea -- click in, select all, copy -- and a download link beside it, so
 * nothing depends on it. That is also why the button starts `hidden`: a
 * button that does nothing with JavaScript off is worse than no button.
 *
 * One listener on `document`, so a draft that arrives later inside an htmx
 * swap needs nothing re-bound -- the same reasoning as sections.js.
 */
(function () {
  /* Show every copy button once the script that makes it work is here. */
  function reveal(root) {
    root.querySelectorAll("button.copy-draft[hidden]").forEach(function (button) {
      button.hidden = false;
    });
  }

  /* The button's neighbouring status line: "Copied." or a way to do it by hand. */
  function say(button, message) {
    var status = button.parentElement && button.parentElement.querySelector(".copy-status");
    if (status) {
      status.textContent = message;
    }
  }

  /* The older route, for a browser without the async clipboard API or a page
     not served over https: select the textarea and ask the browser to copy. */
  function copyBySelecting(textarea) {
    textarea.focus();
    textarea.select();
    try {
      return document.execCommand("copy");
    } catch (e) {
      return false;
    }
  }

  document.addEventListener("click", function (event) {
    var button = event.target instanceof Element && event.target.closest("button.copy-draft");
    if (!button) {
      return;
    }
    var textarea = document.getElementById(button.getAttribute("data-copy-from"));
    if (!textarea) {
      return;
    }
    var fallback = function () {
      say(button, copyBySelecting(textarea) ? "Copied." : "Select the text in the box and copy it.");
    };
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(textarea.value).then(function () {
        say(button, "Copied.");
      }, fallback);
    } else {
      fallback();
    }
  });

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () { reveal(document); });
  } else {
    reveal(document);
  }
  document.addEventListener("htmx:afterSettle", function (event) {
    reveal(event.target instanceof Element ? event.target : document);
  });
})();
