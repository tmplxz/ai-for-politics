import CustomBox from "./CustomBox";

export default function PartyCard({
  party,
  align = "left",
  selectedImageIndex = 0,
}) {
  const selectedImage = party.images[selectedImageIndex];
  const isLeft = align === "left";

  const keypoints = (party.texts || "").split(",").map((point) => point.trim());

  return (
    <CustomBox title={party.metadata.displayName || party.name} align={align}>
      <div
        className={`flex flex-col sm:flex-row items-center gap-4 ${
          isLeft ? "sm:flex-row" : "sm:flex-row-reverse"
        }`}
      >
        <div className="text-black text-center sm:text-left w-full sm:w-1/2 space-y-2">
          {keypoints.length > 0 && keypoints[0] !== "" ? (
            keypoints.map((point, idx) => (
              <div
                key={idx}
                className="bg-white border-l-4 border-primary pl-3 py-1 rounded shadow-sm"
              >
                • {point}
              </div>
            ))
          ) : (
            <p className="text-gray-500">Keine Beschreibung vorhanden.</p>
          )}
        </div>

        {selectedImage ? (
          <div className="rounded-md w-full sm:w-1/2 transition-all duration-500">
            <img
              src={selectedImage}
              alt={`${party.name} Bild ${selectedImageIndex + 1}`}
              className="w-full h-auto object-cover rounded shadow-md"
            />
          </div>
        ) : (
          <div className="w-full sm:w-1/2 h-48 bg-gray-300 flex items-center justify-center rounded-md">
            <span className="text-gray-500">Kein Bild</span>
          </div>
        )}
      </div>
    </CustomBox>
  );
}
