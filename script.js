// Intersection Observer to reveal sections on scroll
const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("visible");
        observer.unobserve(entry.target);
      }
    });
  },
  {
    threshold: 0.15,
  }
);

document.querySelectorAll(".section").forEach((section) => {
  observer.observe(section);
});

// Subtle hover scaling already handled via CSS for cards. For non-card
// sections, keep a very gentle scale on hover to maintain a calm feel.

document.querySelectorAll(".section").forEach((section) => {
  section.addEventListener("mouseenter", () => {
    section.style.transform = "translateY(-2px)";
  });
  section.addEventListener("mouseleave", () => {
    section.style.transform = section.classList.contains("visible")
      ? "translateY(0)"
      : "translateY(28px)";
  });
});
