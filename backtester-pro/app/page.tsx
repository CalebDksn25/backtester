import Image from "next/image";

export default function Home() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-zinc-50 font-sans dark:bg-black">
      <main className="flex min-h-screen w-full max-w-3xl flex-col items-center justify-between py-32 px-16 bg-white dark:bg-black sm:items-start">
        <div className="flex flex-col items-center gap-6 text-center sm:items-start sm:text-left">
          <h1 className="text-5xl font-extrabold leading-tight text-black dark:text-white sm:text-6xl">
            Welcome to Backtester pro, your prompt based backtester.
          </h1>
        </div>
      </main>
    </div>
  );
}
