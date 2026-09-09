document.documentElement.dataset.theme = "clear";
try {
  const savedTheme = localStorage.getItem("hayate-studio-theme");
  if (savedTheme === "clear" || savedTheme === "dark") {
    document.documentElement.dataset.theme = savedTheme;
  }
} catch {
  // Storage can be disabled; Clear remains the default.
}
