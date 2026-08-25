try {
  const savedTheme = localStorage.getItem("hayate-studio-theme");
  if (savedTheme === "clear" || savedTheme === "dark") {
    document.documentElement.dataset.theme = savedTheme;
  }
} catch {
  // Storage can be disabled; Dark remains the safe default.
}
