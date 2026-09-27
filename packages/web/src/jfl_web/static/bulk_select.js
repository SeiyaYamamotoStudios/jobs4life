/* Bulk-select helpers for the applications list -- owner feedback,
 * 2026-09-27: a "select all" checkbox and a running count on the bulk-action
 * button.
 *
 * Enhancement only. Every checkbox here already carries `name` and `value`
 * and points at its form through the `form` attribute, so a batch archive or
 * restore posts correctly with this script blocked -- a person just has to
 * tick each row by hand, and the button always reads its real submitted
 * count from the server's own response (the "Archived N." banner), never
 * from this script.
 *
 * One listener on `document`, delegated, so rows swapped in later by htmx
 * (a status change re-renders its own row) need nothing re-bound.
 */
(function () {
  function rowCheckboxes() {
    return Array.prototype.slice.call(document.querySelectorAll(".row-select"));
  }

  function updateCounts() {
    var checked = rowCheckboxes().filter(function (box) {
      return box.checked;
    }).length;
    document.querySelectorAll("[data-selected-count]").forEach(function (el) {
      el.textContent = String(checked);
    });
  }

  document.addEventListener("change", function (event) {
    var target = event.target;
    if (!(target instanceof Element)) {
      return;
    }
    if (target.matches("[data-select-all]")) {
      rowCheckboxes().forEach(function (box) {
        box.checked = target.checked;
      });
      updateCounts();
      return;
    }
    if (target.matches(".row-select")) {
      updateCounts();
    }
  });

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", updateCounts);
  } else {
    updateCounts();
  }
  document.addEventListener("htmx:afterSettle", updateCounts);
})();
