// tailwind.config.js
/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        primary: "rgb(197, 34, 123)", // #C5227B
        secondary: "rgb(82, 180, 155)", // #52B49B
        third: "rgb(237, 169, 19)", // #EDA913
        whiteish: "rgb(255, 252, 249)", // #FFFCF9
        black: "#000000",
      },
      fontFamily: {
        sans: ["Montserrat", "sans-serif"],
      },
      screens: {
        xs: "100px",
      },
      container: {
        center: true,
        padding: {
          DEFAULT: "1rem",
          sm: "1.5rem",
          lg: "2rem",
        },
      },
    },
  },
  plugins: [],
};
