import { useEffect, useState } from "react";
import Papa from "papaparse";

import parties_metadata from "../../public/parties_metadata.json";
import { useTranslation } from "react-i18next";

export default function EmissionsTable() {
  const { t } = useTranslation();
  const emissions_fname = `${parties_metadata.paths.base}/emissions.csv`
  const [rows, setRows] = useState([]);

  useEffect(() => {
    fetch(emissions_fname)
      .then(res => res.text())
      .then(text => {
        const { data } = Papa.parse(text, { header: true, skipEmptyLines: true });
        // Only keep the desired columns
        const filtered = data.map(row => ({
          experiment_id: row.experiment_id,
          duration: row.duration,
          energy_consumed: row.energy_consumed,
          emissions: row.emissions,
        }));
        setRows(filtered.slice(0, 5)); // show first 4 rows
      });
  }, []);

  // Calculate totals
  const totalDuration = rows.reduce((sum, row) => sum + Number(row.duration), 0);
  const totalEnergy = rows.reduce((sum, row) => sum + Number(row.energy_consumed), 0);
  const totalEmissions = rows.reduce((sum, row) => sum + Number(row.emissions), 0);

  return (
    <div className="p-4">
  <table className="w-full text-xs sm:text-sm md:text-base border-collapse border border-gray-300">
    <thead className="bg-gray-100">
      <tr>
        <th className="px-1 py-1 sm:px-2 sm:py-2 text-left">{t("technik.tab_heading1")}</th>
        <th className="px-1 py-1 sm:px-2 sm:py-2 text-left">{t("technik.tab_heading2")}</th>
        <th className="px-1 py-1 sm:px-2 sm:py-2 text-left">{t("technik.tab_heading3")}</th>
        <th className="px-1 py-1 sm:px-2 sm:py-2 text-left">{t("technik.tab_heading4")}</th>
      </tr>
    </thead>
    <tbody>
      {rows.map((row, i) => (
        <tr key={i} className="border-t border-gray-200">
          <td className="px-1 py-1 sm:px-2 sm:py-2">{t(`technik.tab_${row.experiment_id}`)}</td>
          <td className="px-1 py-1 sm:px-2 sm:py-2">{(Number(row.duration) / 60).toFixed(2)} min</td>
          <td className="px-1 py-1 sm:px-2 sm:py-2">{Number(row.energy_consumed).toFixed(2)} kWh</td>
          <td className="px-1 py-1 sm:px-2 sm:py-2">{Number(row.emissions).toFixed(2)} CO2eq</td>
        </tr>
      ))}
      <tr className="font-bold border-t border-gray-400">
        <td className="px-1 py-1 sm:px-2 sm:py-2">{t("technik.tab_total")}</td>
        <td className="px-1 py-1 sm:px-2 sm:py-2">{(totalDuration / 60).toFixed(2)} min</td>
        <td className="px-1 py-1 sm:px-2 sm:py-2">{totalEnergy.toFixed(2)} kWh</td>
        <td className="px-1 py-1 sm:px-2 sm:py-2">{totalEmissions.toFixed(2)} CO2eq</td>
      </tr>
    </tbody>
  </table>
</div>
  
  );
}
