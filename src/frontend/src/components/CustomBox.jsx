const CustomBox = ({ title, align = "left", children }) => {
  const justify = align === "left" ? "justify-start" : "justify-end";
  const textAlign = align === "left" ? "text-left" : "text-right";

  return (
    <div className="border-2 border-primary rounded-xl min-h-[200px] flex flex-col gap-4 m-4">
      <div className={`text-4xl font-bold text-secondary flex ${justify} pt-4 px-4`}>
        <span className={textAlign}>{title}</span>
      </div>
      <div className="flex flex-col items-center justify-center gap-6 p-4">
        {children}
      </div>
    </div>
  );
};

export default CustomBox;
