import "./App.css";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Layout from "./pages/Layout";
import HomePage from "./pages/HomePage";
import ImpressumPage from "./pages/ImpressumPage";
import PrivacyPage from "./pages/PrivacyPage";
import ProjectPage from "./pages/ProjectPage";
import TechPage from "./pages/TechPage";

function App() {
  return (
    <Router basename="/">
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<HomePage />} />
          <Route path="impressum" element={<ImpressumPage />} />
          <Route path="datenschutz" element={<PrivacyPage />} />
          <Route path="projekt" element={<ProjectPage />} />
          <Route path="technik" element={<TechPage />} />
        </Route>
      </Routes>
    </Router>
  );
}

export default App;
