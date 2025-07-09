let socket = new WebSocket("ws://localhost:8000/events");

socket.onmessage = function(event) {
  switch(event.data){
    case "right":
      console.log("received 'right' event");
      Reveal.right();
      break;
    case "left":
      console.log("received 'left' event");
      Reveal.left();
      break;
    case "rotate":
      console.log("received 'rotate' event");
      const currentSlide = Reveal.getCurrentSlide();

      rotateRotatables(currentSlide);  // defined in helper_methods.js
      break;
    case "pinch":
      console.log("received 'zoom_in' event");

      // increases zoom by 10%
      zoom(10); // `zoom()` is defined in helper_methods.js
      break;
    case "spread":
      console.log("received 'zoom_out' event");

      // decreases zoom by 10%
      zoom(-10); // `zoom()` is defined in helper_methods.js
      break;
    case "swipe_up":
      console.log("received 'swipe_up' event");

      Reveal.up();
      break;
    case "swipe_down":
      console.log("received 'swipe_down' event");

      Reveal.down();
      break;
    case "flip_table":
        console.log("received 'flip_table' event");
        const curSlide = Reveal.getCurrentSlide();
  
        Reveal.shuffle();
        break;
    default:
      console.debug(`unknown message received from server: ${event.data}`);
  }
};
