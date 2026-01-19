import { useState } from "react";
import Disclaimer from "../components/Disclaimer";
import { Slider } from "../components/Slider";
import { useTranslation } from "react-i18next";
import parties_metadata from "../../public/parties_metadata.json";
import Grid from "../components/Grid";
import GridSelector from "../components/GridSelector";

function HomePage() {
  const { t, i18n } = useTranslation();
  // default to "kommu" (kommunalomat view). Use "program" to show program view.
  const [grid, setGrid] = useState("program");

  return (
    <div className="flex flex-col xs:gap-3 items-center justify-center">
      <h1 className="text-primary py-8 md:p-6 text-2xl md:text-3xl font-bold text-center">
        {t("home.second_title")}
      </h1>

      <div className="max-w-5xl w-full self-center sm:px-2 xs:px-8">
        <Slider metadata={parties_metadata} lang={i18n.language} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 items-center border-t-2 px-4 border-secondary mt-8 md:mt-10 w-full">
        <div className="hidden md:block" /> {/* left spacer only on md+ */}
        <h1 className="justify-self-center xs:text-wrap md:text-nowrap text-secondary py-6 md:py-8 text-center whitespace-nowrap">
          {t("home.programs_title")}
        </h1>

        <div className="justify-self-center lg:justify-self-end mt-4 md:mt-0">
          <GridSelector selected={grid} onSelect={setGrid} />
        </div>

      </div>

      <div
        className="w-full lg:max-w-[50%] my-2"
        dangerouslySetInnerHTML={{ __html: t("home.programs_text") }}
        />

      <Grid parties_metadata={parties_metadata} gridKey={grid} />

      <div className="max-w-3xl w-full self-center px-2 md:px-0">
        <Disclaimer />
      </div>
    </div>
  );
}

export default HomePage;
