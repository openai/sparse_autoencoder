import React from "react"

type Props = {
  content: React.ReactNode;
  tooltip: React.ReactNode;
  tooltipStyle?: React.CSSProperties;
};

function useOutsideClickAlerter(ref, fn) {
  React.useEffect(() => {
    /**
     * Alert if clicked on outside of element
     */
    function handleClickOutside(event) {
      if (ref.current && !ref.current.contains(event.target)) { fn() }
    }
    // Bind the event listener
    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      // Unbind the event listener on clean up
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [ref]);
}

const Tooltip: React.FunctionComponent<Props> = (props: Props) => {
  const [state, set_state] = React.useState({ force_open: false });
  const [hover, set_hover] = React.useState(false);
  const wrapperRef = React.useRef(null);
  useOutsideClickAlerter(wrapperRef, () => set_state({ force_open: false }));
  return (
    <span>
    <span ref={wrapperRef} onClick={() => set_state({ force_open: !state.force_open })}
      onMouseOver={() => set_hover(true)}
      onMouseOut={() => setTimeout(() => set_hover(false), 100)}
    >
        {props.content}
      </span>
    <span 
      style={{ ...(props.tooltipStyle || {}), display: (state.force_open || hover) ? 'inline-block' : 'none', position: "absolute", backgroundColor: '#eeeeee', border: '2px solid gray', borderRadius: '2px' }}
    >
        {props.tooltip}
      </span>
    </span>
  );
};


export default Tooltip;
