from IPython.display import HTML, display

from .functools import static_vars

@static_vars(initialized=False)
def notebook_put_into_clipboard(text):

    if not notebook_put_into_clipboard.initialized:

        init = HTML("""
        <script>
        function updateClipboard(newClip) {
          navigator.clipboard.writeText(newClip).then(function() {
            /* clipboard successfully set */
          }, function() {
            /* clipboard write failed */
          });
        }
        </script>""")
        display(init)

    display(HTML(f'<script> updateClipboard("{text}")  </script>'))
