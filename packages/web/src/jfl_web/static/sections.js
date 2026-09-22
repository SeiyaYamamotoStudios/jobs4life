/* Remembering which sections you left open.
 *
 * This file is enhancement and nothing else. `<details>`/`<summary>` folds and
 * unfolds with no JavaScript at all; block this script and every section still
 * works, it just stops being remembered between visits. Nothing here changes
 * what a page says, and nothing here is on the path of anything that costs
 * money.
 *
 * htmx cannot do this on its own for one reason: the value to send is the
 * element's `open` *property* after the toggle, and a property is not an
 * attribute htmx can read out of the DOM. So this reads the one boolean and
 * hands the request straight back to htmx -- the post, the headers and the
 * error handling stay htmx's, and there is no second way of talking to the
 * server in this codebase.
 *
 * The listener is registered in the capture phase, deliberately. `toggle` does
 * not bubble, so a listener on `document` would never see it on the way up --
 * but every event, bubbling or not, passes through the capture phase on its way
 * down. That is what makes one listener enough, including for sections that
 * arrive later inside an htmx swap; there is nothing to re-bind after a swap
 * and therefore nothing to forget to re-bind.
 */
document.addEventListener(
  "toggle",
  function (event) {
    var details = event.target;
    if (!(details instanceof Element) || !details.matches("details[data-section]")) {
      return;
    }
    /* Same token every form on the page carries. Without it the post is
       rejected, which is correct -- this is a state-changing request like any
       other -- so a page with no token (signed out) simply does not post. */
    var token = document.body.getAttribute("data-csrf");
    if (!token) {
      return;
    }
    htmx.ajax("POST", "/ui/sections", {
      swap: "none",
      values: {
        csrf_token: token,
        section: details.getAttribute("data-section"),
        open: details.open ? "true" : "false",
        default_open: details.getAttribute("data-default-open")
      }
    });
  },
  true
);
